"""Async UART receiver for ECG DAQ system."""

import asyncio
import struct
import time
from typing import AsyncGenerator, Optional, Callable
from collections import deque
import serial
import serial_asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import threading
import numpy as np

from ..core.config import Config
from ..core.models import Packet, Sample
from ..core.exceptions import DataAcquisitionError, ProtocolError
from .packet_parser import PacketParser


class RingBuffer:
    """High-performance ring buffer for continuous data streaming."""
    
    def __init__(self, size: int):
        """Initialize ring buffer with specified size."""
        self.size = size
        self.buffer = bytearray(size)
        self.head = 0  # Write position
        self.tail = 0  # Read position
        self.count = 0  # Number of bytes in buffer
        self.lock = threading.Lock()
    
    def write(self, data: bytes) -> int:
        """Write data to ring buffer. Returns number of bytes written."""
        if not data:
            return 0
        
        with self.lock:
            data_len = len(data)
            available_space = self.size - self.count
            
            if data_len > available_space:
                # Buffer overflow - advance tail to make space
                overflow = data_len - available_space
                self.tail = (self.tail + overflow) % self.size
                self.count -= overflow
                print(f"WARNING: Ring buffer overflow, lost {overflow} bytes")
            
            # Write data in two parts if it wraps around
            bytes_to_end = self.size - self.head
            if data_len <= bytes_to_end:
                # Data fits without wrapping
                self.buffer[self.head:self.head + data_len] = data
                self.head = (self.head + data_len) % self.size
            else:
                # Data wraps around
                self.buffer[self.head:] = data[:bytes_to_end]
                remaining = data_len - bytes_to_end
                self.buffer[:remaining] = data[bytes_to_end:]
                self.head = remaining
            
            self.count += data_len
            return data_len
    
    def read(self, max_bytes: int) -> bytes:
        """Read up to max_bytes from ring buffer."""
        with self.lock:
            if self.count == 0:
                return b''
            
            bytes_to_read = min(max_bytes, self.count)
            
            # Read data in two parts if it wraps around
            bytes_to_end = self.size - self.tail
            if bytes_to_read <= bytes_to_end:
                # Data doesn't wrap
                data = bytes(self.buffer[self.tail:self.tail + bytes_to_read])
                self.tail = (self.tail + bytes_to_read) % self.size
            else:
                # Data wraps around
                data = bytes(self.buffer[self.tail:]) + bytes(self.buffer[:bytes_to_read - bytes_to_end])
                self.tail = bytes_to_read - bytes_to_end
            
            self.count -= bytes_to_read
            return data
    
    def peek(self, max_bytes: int) -> bytes:
        """Peek at data without removing it from buffer."""
        with self.lock:
            if self.count == 0:
                return b''
            
            bytes_to_read = min(max_bytes, self.count)
            
            # Read data in two parts if it wraps around
            bytes_to_end = self.size - self.tail
            if bytes_to_read <= bytes_to_end:
                # Data doesn't wrap
                return bytes(self.buffer[self.tail:self.tail + bytes_to_read])
            else:
                # Data wraps around
                return bytes(self.buffer[self.tail:]) + bytes(self.buffer[:bytes_to_read - bytes_to_end])
    
    def discard(self, num_bytes: int) -> None:
        """Discard num_bytes from the front of the buffer."""
        with self.lock:
            bytes_to_discard = min(num_bytes, self.count)
            self.tail = (self.tail + bytes_to_discard) % self.size
            self.count -= bytes_to_discard
    
    @property
    def available_bytes(self) -> int:
        """Get number of bytes available to read."""
        return self.count
    
    @property
    def free_space(self) -> int:
        """Get number of bytes of free space."""
        return self.size - self.count


def parse_packet_worker(packet_data: bytes, config_dict: dict) -> dict:
    """Worker function for parsing packets in separate process."""
    try:
        # Recreate config and parser in worker process
        config = Config(**config_dict)
        parser = PacketParser(config)
        
        # Parse the packet
        packet = parser.parse_packet(packet_data)
        
        # Convert to dictionary for IPC with high precision
        return {
            'success': True,
            'packet_type': packet.packet_type,
            'packet_number': packet.packet_number,
            'samples': [
                {
                    'channels': [float(ch) for ch in sample.channels],  # Ensure high precision floats
                    'saturation': sample.saturation,
                    'pace_detected': sample.pace_detected,
                    'lead_off_bits': sample.lead_off_bits,
                    'sign_bits': sample.sign_bits,
                    'raw_metadata': list(sample.raw_metadata) if sample.raw_metadata else None
                }
                for sample in packet.samples
            ],
            'raw_data_size': packet.raw_data_size
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


class SerialProtocol(asyncio.Protocol):
    """Protocol for handling serial data reception."""
    
    def __init__(self, data_callback):
        """Initialize protocol with data callback."""
        self.data_callback = data_callback
        self.transport = None
    
    def connection_made(self, transport):
        """Called when connection is established."""
        self.transport = transport
    
    def data_received(self, data):
        """Called when data is received."""
        if self.data_callback:
            self.data_callback(data)
    
    def connection_lost(self, exc):
        """Called when connection is lost."""
        if exc:
            print(f"Serial connection lost: {exc}")


class AsyncUARTReceiver:
    """Async UART receiver with efficient packet processing."""
    
    def __init__(self, config: Config):
        """
        Initialize async UART receiver.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.parser = PacketParser(config)
        self._running = False
        
        # High-performance ring buffer (16MB default)
        buffer_size = max(config.max_buffer_size, 16 * 1024 * 1024)
        self._ring_buffer = RingBuffer(buffer_size)
        
        # Async processing queues
        self._data_queue = asyncio.Queue(maxsize=1000)  # Raw data chunks
        self._packet_queue = asyncio.Queue(maxsize=500)  # Parsed packets
        
        # Statistics
        self._packet_count = 0
        self._error_count = 0
        self._start_time = None
        self._bytes_received = 0
        self._bytes_processed = 0
        
        # Packet continuity tracking
        self._last_packet_number = None
        self._missing_packets = 0
        self._duplicate_packets = 0
        
        # Processing executors
        self._thread_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="packet-thread")
        self._process_executor = ProcessPoolExecutor(max_workers=min(4, multiprocessing.cpu_count()))
        
        # Config dict for worker processes
        self._config_dict = config.dict()
        
        # Callbacks
        self._packet_callback: Optional[Callable[[Packet], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._status_callback: Optional[Callable[[dict], None]] = None
        
    async def start(self) -> None:
        """Start the UART receiver."""
        if self._running:
            return
            
        self._running = True
        self._start_time = time.time()
        
        try:
            # Create serial connection with custom protocol
            def protocol_factory():
                return SerialProtocol(self._on_data_received)
            
            transport, protocol = await serial_asyncio.create_serial_connection(
                asyncio.get_event_loop(),
                protocol_factory,
                self.config.uart.port,
                baudrate=self.config.uart.baudrate,
                bytesize=self.config.uart.bytesize,
                parity=self.config.uart.parity,
                stopbits=self.config.uart.stopbits
            )
            
            self._transport = transport
            self._protocol = protocol
            
            # Start background tasks
            asyncio.create_task(self._read_loop())
            asyncio.create_task(self._packet_extraction_loop())
            asyncio.create_task(self._packet_parsing_loop())
            asyncio.create_task(self._status_loop())
            
        except Exception as e:
            self._running = False
            raise DataAcquisitionError(f"Failed to start UART receiver: {e}")
    
    async def stop(self) -> None:
        """Stop the UART receiver."""
        self._running = False
        
        if hasattr(self, '_transport'):
            self._transport.close()
        
        # Shutdown executors
        if hasattr(self, '_thread_executor'):
            self._thread_executor.shutdown(wait=False)
        if hasattr(self, '_process_executor'):
            self._process_executor.shutdown(wait=False)
    
    def set_packet_callback(self, callback: Callable[[Packet], None]) -> None:
        """Set callback for received packets."""
        self._packet_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors."""
        self._error_callback = callback
    
    def set_status_callback(self, callback: Callable[[dict], None]) -> None:
        """Set callback for status updates."""
        self._status_callback = callback
    
    def _on_data_received(self, data: bytes) -> None:
        """Handle data received from serial port."""
        try:
            # Write directly to ring buffer (thread-safe)
            self._bytes_received += len(data)
            bytes_written = self._ring_buffer.write(data)
            
            if bytes_written < len(data):
                print(f"WARNING: Ring buffer full, lost {len(data) - bytes_written} bytes")
            
        except Exception as e:
            if self._error_callback:
                self._error_callback(e)
            else:
                print(f"Data reception error: {e}")
    
    async def _read_loop(self) -> None:
        """Main read loop - just monitors connection health."""
        while self._running:
            try:
                # Check connection health
                if hasattr(self, '_transport') and self._transport.is_closing():
                    print("WARNING: Transport is closing")
                    break
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self._error_count += 1
                if self._error_callback:
                    self._error_callback(e)
                else:
                    print(f"Connection monitoring error: {e}")
                
                await asyncio.sleep(1.0)
    
    async def _packet_extraction_loop(self) -> None:
        """High-performance packet extraction from ring buffer."""
        start_bytes = struct.pack(">I", self.config.protocol.start_word)
        stop_bytes = struct.pack(">I", self.config.protocol.stop_word)
        
        extraction_buffer = bytearray()
        
        while self._running:
            try:
                # Read chunk from ring buffer
                chunk = self._ring_buffer.read(8192)  # Read 8KB chunks
                if not chunk:
                    await asyncio.sleep(0.001)  # Short sleep if no data
                    continue
                
                extraction_buffer.extend(chunk)
                self._bytes_processed += len(chunk)
                
                # Extract complete packets
                await self._extract_packets_from_buffer(extraction_buffer, start_bytes, stop_bytes)
                
                # Keep buffer size reasonable
                if len(extraction_buffer) > 65536:  # 64KB limit
                    # Keep only the last 32KB
                    extraction_buffer = extraction_buffer[-32768:]
                
            except Exception as e:
                self._error_count += 1
                print(f"Packet extraction error: {e}")
                await asyncio.sleep(0.01)
    
    async def _extract_packets_from_buffer(self, buffer: bytearray, start_bytes: bytes, stop_bytes: bytes) -> None:
        """Extract packets from buffer and queue for parsing."""
        while True:
            # Find start marker
            start_idx = buffer.find(start_bytes)
            if start_idx < 0:
                # No start marker found, keep only last few bytes in case of partial marker
                if len(buffer) > 8:
                    buffer[:] = buffer[-4:]
                break
            
            # Remove data before start marker
            if start_idx > 0:
                del buffer[:start_idx]
                start_idx = 0
            
            # Find stop marker
            stop_idx = buffer.find(stop_bytes, 4)  # Start search after start marker
            if stop_idx < 0:
                # No complete packet yet
                break
            
            # Extract packet data (excluding start/stop markers)
            packet_data = bytes(buffer[4:stop_idx])  # Skip start marker
            
            # Queue packet for parsing (non-blocking)
            try:
                self._packet_queue.put_nowait(packet_data)
            except asyncio.QueueFull:
                print("WARNING: Packet parsing queue full, dropping packet")
                self._error_count += 1
            
            # Remove processed data
            del buffer[:stop_idx + 4]  # Include stop marker
    
    async def _packet_parsing_loop(self) -> None:
        """High-performance async packet parsing using process pool."""
        pending_futures = set()
        
        while self._running or pending_futures:
            try:
                # Get packet data from queue
                try:
                    packet_data = await asyncio.wait_for(self._packet_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if not self._running:
                        break
                    continue
                
                # Submit to process pool for parsing
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(
                    self._process_executor,
                    parse_packet_worker,
                    packet_data,
                    self._config_dict
                )
                
                # Add callback to handle result
                future.add_done_callback(lambda f: asyncio.create_task(self._handle_parsed_packet(f)))
                pending_futures.add(future)
                
                # Clean up completed futures
                completed = {f for f in pending_futures if f.done()}
                pending_futures -= completed
                
                # Limit number of pending parsing operations
                if len(pending_futures) > 50:
                    # Wait for some to complete
                    done, pending_futures = await asyncio.wait(
                        pending_futures, 
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=0.01
                    )
                
            except Exception as e:
                self._error_count += 1
                print(f"Packet parsing loop error: {e}")
                await asyncio.sleep(0.01)
    
    async def _handle_parsed_packet(self, future) -> None:
        """Handle result from packet parsing."""
        try:
            result = future.result()
            
            if result['success']:
                # Reconstruct packet from dict
                samples = []
                for sample_dict in result['samples']:
                    sample = Sample(
                        channels=sample_dict['channels'],
                        saturation=sample_dict['saturation'],
                        pace_detected=sample_dict['pace_detected'],
                        lead_off_bits=sample_dict['lead_off_bits'],
                        sign_bits=sample_dict['sign_bits'],
                        raw_metadata=bytes(sample_dict['raw_metadata']) if sample_dict['raw_metadata'] else None
                    )
                    samples.append(sample)
                
                packet = Packet(
                    packet_type=result['packet_type'],
                    packet_number=result['packet_number'],
                    samples=samples,
                    raw_data_size=result['raw_data_size']
                )
                
                # Check continuity and handle packet
                await self._handle_packet_continuity(packet)
                
            else:
                # Handle parsing error
                self._error_count += 1
                print(f"Packet parsing failed: {result['error_type']}: {result['error']}")
                
        except Exception as e:
            self._error_count += 1
            print(f"Error handling parsed packet: {e}")
    
    async def _handle_packet_continuity(self, packet: Packet) -> None:
        """Handle packet continuity checking and callbacks."""
        try:
            # Check packet continuity
            if self._last_packet_number is not None:
                expected_number = (self._last_packet_number + 1) % 65536  # Handle wraparound
                if packet.packet_number != expected_number:
                    if packet.packet_number > expected_number:
                        # Missing packets
                        missed = packet.packet_number - expected_number
                        self._missing_packets += missed
                        print(f"WARNING: Missing {missed} packets. Expected {expected_number}, got {packet.packet_number}")
                    elif packet.packet_number < expected_number:
                        # Could be wraparound or duplicate
                        if expected_number - packet.packet_number > 32768:  # Likely wraparound
                            # This is normal wraparound, no action needed
                            pass
                        else:
                            # Duplicate packet
                            self._duplicate_packets += 1
                            print(f"WARNING: Duplicate packet {packet.packet_number}")
                            return  # Skip duplicate packets
            
            self._last_packet_number = packet.packet_number
            self._packet_count += 1
            
            # Call packet callback if set
            if self._packet_callback:
                self._packet_callback(packet)
                
        except Exception as e:
            self._error_count += 1
            print(f"Packet continuity handling error: {e}")
    
    # Legacy compatibility methods - now unused
    async def _process_data(self, data: bytes, start_bytes: bytes, stop_bytes: bytes) -> None:
        """Legacy method - now handled by ring buffer system."""
        pass
    
    async def _status_loop(self) -> None:
        """Status update loop."""
        while self._running:
            try:
                status = self._get_status()
                if self._status_callback:
                    self._status_callback(status)
                
                await asyncio.sleep(1.0)  # Update status every second
                
            except Exception as e:
                print(f"Status update error: {e}")
                await asyncio.sleep(1.0)
    
    def _get_status(self) -> dict:
        """Get current status."""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        # Calculate throughput
        bytes_per_sec = self._bytes_received / uptime if uptime > 0 else 0
        packets_per_sec = self._packet_count / uptime if uptime > 0 else 0
        
        return {
            'is_running': self._running,
            'packets_received': self._packet_count,
            'errors_count': self._error_count,
            'missing_packets': self._missing_packets,
            'duplicate_packets': self._duplicate_packets,
            'last_packet_number': self._last_packet_number,
            'bytes_received': self._bytes_received,
            'bytes_processed': self._bytes_processed,
            'bytes_per_second': bytes_per_sec,
            'packets_per_second': packets_per_sec,
            'uptime': uptime,
            'ring_buffer_size': self._ring_buffer.available_bytes,
            'ring_buffer_utilization': self._ring_buffer.available_bytes / self._ring_buffer.size,
            'packet_queue_size': self._packet_queue.qsize(),
            'data_queue_size': self._data_queue.qsize()
        }
    
    async def packets(self) -> AsyncGenerator[Packet, None]:
        """Async generator for packets."""
        packet_queue = asyncio.Queue()
        
        def packet_handler(packet: Packet):
            asyncio.create_task(packet_queue.put(packet))
        
        self.set_packet_callback(packet_handler)
        
        try:
            while self._running:
                try:
                    packet = await asyncio.wait_for(packet_queue.get(), timeout=1.0)
                    yield packet
                except asyncio.TimeoutError:
                    continue
        finally:
            # Clean up
            pass
    
    @property
    def is_running(self) -> bool:
        """Check if receiver is running."""
        return self._running
    
    @property
    def packet_count(self) -> int:
        """Get total packet count."""
        return self._packet_count
    
    @property
    def error_count(self) -> int:
        """Get total error count."""
        return self._error_count


# Legacy synchronous receiver for backward compatibility
class UARTReceiver:
    """Legacy synchronous UART receiver."""
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.1,
                 plotter=None):
        """Initialize legacy receiver."""
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )
        self._buf = bytearray()
        self.plotter = plotter
    
    def close(self):
        """Close serial connection."""
        try:
            self.ser.close()
        except Exception:
            pass
    
    def packets(self):
        """Legacy packet generator."""
        # This is a simplified version for backward compatibility
        # In production, use AsyncUARTReceiver
        pass
