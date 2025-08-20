#!/usr/bin/env python3
"""Real-time ECG monitoring from hardware via serial port."""

import asyncio
import time
import signal
import sys
from pathlib import Path

# Imports are now relative since we're inside the package

from ..core.config import Config, CRCMethod
from ..core.exceptions import ECGDAQError
from ..data_acquisition.uart_receiver import AsyncUARTReceiver
import csv
import json
from datetime import datetime


class RealTimeMonitor:
    """Real-time ECG monitor with hardware data acquisition."""
    
    def __init__(self, config: Config):
        """
        Initialize real-time monitor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.receiver = None
        self.data_file = None
        self.csv_writer = None
        self.running = False
        self.packet_count = 0
        self.last_saved_packet_number = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False
    
    async def start(self):
        """Start the real-time monitoring system."""
        print("Starting Real-time ECG Data Logger...")
        print(f"Port: {self.config.uart.port}")
        print(f"Baudrate: {self.config.uart.baudrate}")
        print(f"Sample Rate: {self.config.ecg.sample_rate} Hz")
        print(f"CRC Method: {self.config.protocol.crc_method.value}")
        
        # Create data file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ecg_data_{timestamp}.csv"
        print(f"Data will be saved to: {filename}")
        print("-" * 50)
        
        try:
            # Create and start UART receiver
            self.receiver = AsyncUARTReceiver(self.config)
            
            # Set up callbacks
            self.receiver.set_error_callback(self._on_error)
            self.receiver.set_status_callback(self._on_status)
            
            # Start receiver
            await self.receiver.start()
            print("UART receiver started successfully.")
            
            # Setup data file
            self.data_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.data_file)
            
            # Write CSV header with units (millivolts)
            header = ['time_seconds', 'packet_number', 'packet_type', 'sample_index']
            for i in range(8):  # 8 channels
                header.append(f'channel_{i+1}')  # Values are in millivolts
            header.extend(['saturation', 'pace_detected', 'lead_off_bits', 'sign_bits'])
            self.csv_writer.writerow(header)
            
            print("Data logging started successfully.")
            
            # Set packet callback to save data
            self.receiver.set_packet_callback(self._on_packet)
            
            # Start monitoring
            self.running = True
            await self._monitor_loop()
            
        except Exception as e:
            print(f"Failed to start monitoring: {e}")
            raise
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        print("Real-time data logging active. Press Ctrl+C to stop.")
        
        try:
            # Start data collection in background
            data_task = asyncio.create_task(self._collect_data())
            
            # Keep the loop running until stopped
            while self.running:
                await asyncio.sleep(1.0)  # Check every second
            
        except KeyboardInterrupt:
            print("\nData logging interrupted by user.")
        except Exception as e:
            print(f"Monitoring error: {e}")
        finally:
            # Cleanup
            self.running = False
            if not data_task.done():
                data_task.cancel()
                try:
                    await data_task
                except asyncio.CancelledError:
                    pass
    
    async def _collect_data(self):
        """Collect data from UART receiver."""
        try:
            async for packet in self.receiver.packets():
                if not self.running:
                    break
                
                # Save packet data to CSV
                if packet.samples:
                    self._save_packet_data(packet)
                
                # Print packet info (optional, for debugging)
                if packet.packet_number % 100 == 0:  # Print every 100th packet
                    # Show lead-off and sign bit interpretation for first sample
                    first_sample = packet.samples[0]
                    lead_off_channels = []
                    sign_channels = []
                    
                    for i in range(8):
                        channel_num = i + 1
                        
                        # Lead-off: bit i = channel (i+1) - assuming same as before
                        lead_off = bool((first_sample.lead_off_bits >> i) & 0x01)
                        lead_off_channels.append(f"Ch{channel_num}:{'OFF' if lead_off else 'OK'}")
                        
                        # Sign: bit 7 = Channel 1, bit 0 = Channel 8
                        bit_position = 7 - i  # Channel 1 uses bit 7, Channel 8 uses bit 0
                        sign = bool((first_sample.sign_bits >> bit_position) & 0x01)
                        sign_channels.append(f"Ch{channel_num}:{'NEG' if sign else 'POS'}")
                    
                    print(f"Saved packet {packet.packet_number}: "
                          f"{len(packet.samples)} samples (expected 50)")
                    print(f"  Lead-off: {' '.join(lead_off_channels[:4])}")
                    print(f"            {' '.join(lead_off_channels[4:])}")
                    print(f"  Signs:    {' '.join(sign_channels[:4])}")
                    print(f"            {' '.join(sign_channels[4:])}")
                
        except Exception as e:
            print(f"Data collection error: {e}")
    
    def _save_packet_data(self, packet):
        """Save packet data to CSV file."""
        try:
            # Use agreed sample rate from configuration (Tx and Rx must agree on this)
            sample_rate = self.config.ecg.sample_rate
            
            # Calculate time base: each packet with 50 samples represents (50/sample_rate) seconds
            packet_duration = len(packet.samples) / sample_rate  # e.g., 50/500 = 0.1s = 100ms
            
            # Calculate base time for this packet (assuming continuous timing)
            base_time = packet.packet_number * packet_duration  # seconds from start
            
            # Check for missing packets and create placeholder entries
            if self.last_saved_packet_number is not None:
                expected_number = (self.last_saved_packet_number + 1) % 65536
                if packet.packet_number != expected_number and packet.packet_number > expected_number:
                    # Insert placeholder rows for missing packets
                    missing_count = packet.packet_number - expected_number
                    if missing_count < 100:  # Only fill small gaps to avoid huge files
                        print(f"Filling {missing_count} missing packets in CSV...")
                        for missing_num in range(expected_number, packet.packet_number):
                            missing_base_time = missing_num * packet_duration
                            self._save_missing_packet_placeholder(missing_num, missing_base_time, sample_rate)
            
            # Write each sample in the packet
            for sample_idx, sample in enumerate(packet.samples):
                # Calculate precise timestamp for this sample
                sample_time = base_time + (sample_idx / sample_rate)
                
                row = [
                    f"{sample_time:.6f}",  # Time in seconds from start (6 decimal precision)
                    packet.packet_number,
                    packet.packet_type,
                    sample_idx
                ]
                
                # Add channel data with high precision (pad with 0 if fewer than 8 channels)
                channels = sample.channels[:8]  # Take first 8 channels
                while len(channels) < 8:
                    channels.append(0.0)
                
                # Format channels with high precision (6 decimal places for µV precision)
                formatted_channels = [f"{ch:.6f}" for ch in channels]
                row.extend(formatted_channels)
                
                # Add sample metadata
                row.extend([
                    sample.saturation,
                    sample.pace_detected,
                    sample.lead_off_bits,
                    sample.sign_bits
                ])
                
                self.csv_writer.writerow(row)
            
            self.last_saved_packet_number = packet.packet_number
            
            # Flush data to disk periodically
            if packet.packet_number % 10 == 0:
                self.data_file.flush()
                
        except Exception as e:
            print(f"Error saving packet data: {e}")
    
    def _save_missing_packet_placeholder(self, packet_number, base_time, sample_rate):
        """Save placeholder data for missing packets."""
        # Create placeholder rows for each sample in the missing packet (50 samples)
        for sample_idx in range(50):  # 50 samples per packet
            sample_time = base_time + (sample_idx / sample_rate)
            row = [
                f"{sample_time:.6f}",  # Time in seconds
                packet_number,
                0,  # packet_type
                sample_idx
            ]
            
            # Add NaN for all channels with consistent formatting
            for i in range(8):
                row.append("NaN")
            
            # Add placeholder metadata
            row.extend([False, False, 0, 0])  # saturation, pace_detected, lead_off_bits, sign_bits
            
            self.csv_writer.writerow(row)
    
    def _on_packet(self, packet):
        """Handle received packet."""
        # This is called from the UART receiver thread
        # Data saving is handled in the async loop
        pass
    
    def _on_error(self, error):
        """Handle errors from UART receiver."""
        print(f"UART Error: {error}")
    
    def _on_status(self, status):
        """Handle status updates from UART receiver."""
        if status['packets_received'] % 100 == 0:  # Print every 100 packets
            print(f"Status: {status['packets_received']} packets "
                  f"({status['packets_per_second']:.1f}/s), "
                  f"{status['errors_count']} errors, "
                  f"{status['missing_packets']} missing, "
                  f"{status['duplicate_packets']} duplicates, "
                  f"Buffer: {status['ring_buffer_size']/1024:.1f}KB "
                  f"({status['ring_buffer_utilization']*100:.1f}%), "
                  f"Queues: {status['packet_queue_size']}/{status['data_queue_size']}, "
                  f"Throughput: {status['bytes_per_second']/1024:.1f} KB/s, "
                  f"Uptime: {status['uptime']:.1f}s")
    
    async def stop(self):
        """Stop the monitoring system."""
        print("Stopping data logging system...")
        self.running = False
        
        if self.data_file:
            self.data_file.close()
            print("Data file closed.")
        
        if self.receiver:
            await self.receiver.stop()
        
        print("Data logging system stopped.")


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time ECG Data Logger")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM5, /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    parser.add_argument("--sample-rate", type=float, default=500.0, help="Expected sample rate (Hz)")
    parser.add_argument("--crc-method", default="sum", 
                       choices=["sum", "maxim", "ccitt", "dallas", "sae_j1850", "rohc", 
                               "crc16_ccitt", "crc16_modbus", "crc16_xmodem", "crc32"],
                       help="CRC method to use")
    parser.add_argument("--no-crc", action="store_true", help="Disable CRC validation")
    parser.add_argument("--little-endian", action="store_true", help="Use little-endian for channels")
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config.create_default(args.port)
        config.uart.baudrate = args.baudrate
        config.ecg.sample_rate = args.sample_rate
        config.protocol.crc_method = CRCMethod(args.crc_method)
        config.protocol.enable_crc = not args.no_crc
        
        if args.little_endian:
            config.protocol.endianness = "little"
        
        # Create and start monitor
        monitor = RealTimeMonitor(config)
        await monitor.start()
        
    except KeyboardInterrupt:
        print("\nData logging stopped by user.")
    except Exception as e:
        print(f"Data logging failed: {e}")
        sys.exit(1)
    finally:
        if 'monitor' in locals():
            await monitor.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped.")
    except Exception as e:
        print(f"Program failed: {e}")
        sys.exit(1)
