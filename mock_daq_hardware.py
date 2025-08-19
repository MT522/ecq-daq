#!/usr/bin/env python3
"""
Mock DAQ Hardware Simulator using AAMI EC13 Test Waveforms

This script simulates your ECG DAQ hardware by reading AAMI EC13 test waveforms
and transmitting them over a virtual serial port in your exact packet format.

Based on PhysioNet AAMI EC13 Test Waveforms:
https://www.physionet.org/content/aami-ec13/1.0.0/

Packet Structure:
4B STX (0x11223344) + 1B Packet Type + 4B Packet Num + 2B Data Len + 
19×50B Data + 1B CRC + 4B ETX (0xAABBCCDD)

Usage:
    python mock_daq_hardware.py --port COM6 --waveform aami3a
    python mock_daq_hardware.py --port /dev/ttyUSB1 --waveform aami3b --rate 1000
"""

import argparse
import struct
import time
import threading
import serial
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple
import sys
from scipy import signal


class WaveformReader:
    """Reads AAMI EC13 waveform files (.hea/.dat format)."""
    
    def __init__(self, base_path: str, record_name: str, target_sample_rate: float = 500.0):
        """
        Initialize waveform reader.
        
        Args:
            base_path: Path to directory containing waveform files
            record_name: Name of the record (e.g., 'aami3a')
            target_sample_rate: Target sample rate for resampling (default: 500Hz)
        """
        self.base_path = Path(base_path)
        self.record_name = record_name
        self.header_file = self.base_path / f"{record_name}.hea"
        self.data_file = self.base_path / f"{record_name}.dat"
        
        # Waveform parameters
        self.original_sample_rate = 720  # Hz (from AAMI EC13 spec)
        self.target_sample_rate = target_sample_rate
        self.resolution = 12    # bits
        self.samples_count = 0
        self.gain = 1.0
        self.baseline = 0
        
        self._read_header()
        self._load_data()
        self._resample_data()
    
    def _read_header(self):
        """Read the .hea header file."""
        try:
            with open(self.header_file, 'r') as f:
                lines = f.readlines()
            
            # Parse first line: record_name samples sample_rate
            first_line = lines[0].strip().split()
            if len(first_line) >= 3:
                self.samples_count = int(first_line[1])
                self.original_sample_rate = int(first_line[2])
            
            print(f"Loaded {self.record_name}: {self.samples_count} samples at {self.original_sample_rate} Hz")
            
        except Exception as e:
            print(f"Error reading header {self.header_file}: {e}")
            # Use defaults
            self.samples_count = 60000  # ~83 seconds at 720 Hz
    
    def _load_data(self):
        """Load the .dat binary data file."""
        try:
            # AAMI EC13 files are 12-bit data, typically stored as 16-bit integers
            with open(self.data_file, 'rb') as f:
                data_bytes = f.read()
            
            # Convert to 16-bit integers (little-endian)
            num_samples = len(data_bytes) // 2
            self.data = struct.unpack(f'<{num_samples}h', data_bytes)
            
            print(f"Loaded {len(self.data)} data points from {self.data_file}")
            
            # Convert to millivolts (approximate scaling for ECG)
            # AAMI test signals are typically ±5mV range
            max_val = max(abs(min(self.data)), max(self.data))
            if max_val > 0:
                self.scale_factor = 5.0 / max_val  # Scale to ±5mV
            else:
                self.scale_factor = 1.0
                
        except Exception as e:
            print(f"Error loading data {self.data_file}: {e}")
            # Generate synthetic data
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic ECG data if file loading fails."""
        print("Generating synthetic ECG data...")
        duration = 60.0  # 60 seconds
        self.samples_count = int(duration * self.sample_rate)
        
        # Generate synthetic ECG-like waveform
        t = np.linspace(0, duration, self.samples_count)
        
        # Basic ECG components
        heart_rate = 75  # BPM
        rr_interval = 60.0 / heart_rate
        
        ecg = np.zeros_like(t)
        for beat_time in np.arange(0, duration, rr_interval):
            # Simple QRS complex approximation
            for i, time_point in enumerate(t):
                dt = time_point - beat_time
                if 0 <= dt <= 0.1:  # QRS duration ~100ms
                    if 0.02 <= dt <= 0.08:  # R wave
                        ecg[i] += 1.0 * np.exp(-((dt - 0.05) / 0.01) ** 2)
                    elif dt <= 0.02:  # Q wave
                        ecg[i] -= 0.2 * np.exp(-((dt - 0.01) / 0.005) ** 2)
                    elif dt >= 0.08:  # S wave
                        ecg[i] -= 0.3 * np.exp(-((dt - 0.09) / 0.005) ** 2)
        
        # Convert to integer values similar to real data
        self.data = (ecg * 1000).astype(int)
        self.scale_factor = 0.001  # mV conversion
    
    def _resample_data(self):
        """Resample data from original rate to target rate."""
        if len(self.data) == 0 or self.original_sample_rate == self.target_sample_rate:
            print(f"No resampling needed (target rate: {self.target_sample_rate} Hz)")
            return
        
        print(f"Resampling from {self.original_sample_rate} Hz to {self.target_sample_rate} Hz...")
        
        # Calculate resampling ratio
        resample_ratio = self.target_sample_rate / self.original_sample_rate
        new_length = int(len(self.data) * resample_ratio)
        
        # Use scipy.signal.resample for high-quality resampling
        resampled_data = signal.resample(self.data, new_length)
        
        # Convert back to integer
        self.data = resampled_data.astype(int)
        self.samples_count = len(self.data)
        
        print(f"Resampled to {len(self.data)} samples at {self.target_sample_rate} Hz")
    
    def get_sample(self, index: int) -> float:
        """Get a single sample value in millivolts."""
        if len(self.data) == 0:
            return 0.0
        
        sample_index = index % len(self.data)
        raw_value = self.data[sample_index]
        return raw_value * self.scale_factor


class MockDAQHardware:
    """Mock DAQ hardware that transmits ECG data in your packet format."""
    
    def __init__(self, port: str, waveform_path: str, waveform_name: str, 
                 sample_rate: float = 500.0, packet_rate: float = 10.0):
        """
        Initialize mock DAQ hardware.
        
        Args:
            port: Serial port to transmit on
            waveform_path: Path to AAMI waveform files
            waveform_name: Name of waveform to use
            sample_rate: ECG sample rate in Hz
            packet_rate: Packet transmission rate in Hz
        """
        self.port = port
        self.sample_rate = sample_rate
        self.packet_rate = packet_rate
        self.running = False
        
        # Load waveform data
        self.waveform = WaveformReader(waveform_path, waveform_name, sample_rate)
        
        # Packet format constants
        self.STX = 0x11223344
        self.ETX = 0xAABBCCDD
        self.SAMPLES_PER_PACKET = 50
        self.SAMPLE_SIZE = 19  # 3B metadata + 16B channels
        
        # State tracking
        self.packet_number = 0
        self.sample_index = 0
        
        # Serial connection
        self.serial_conn = None
        
    def connect(self):
        """Connect to serial port."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=115200,
                timeout=1.0,
                bytesize=8,
                parity='N',
                stopbits=1
            )
            print(f"Connected to {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from serial port")
    
    def _calculate_crc(self, data: bytes) -> int:
        """Calculate simple checksum CRC."""
        return sum(data) & 0xFF
    
    def _create_sample(self, sample_idx: int) -> bytes:
        """Create a single 19-byte sample."""
        # Get ECG value for primary channel (Channel 1)
        ecg_value = self.waveform.get_sample(sample_idx)
        
        # Convert mV to LSB (10µV per LSB)
        lsb_value = int(ecg_value / 0.01)  # mV to LSB conversion
        
        # Clamp to 16-bit range
        lsb_value = max(-32768, min(32767, lsb_value))
        
        # Create 8 channels with variations
        channels = []
        for i in range(8):
            if i == 0:  # Channel 1 - primary ECG
                channels.append(abs(lsb_value))  # Use absolute value
            elif i == 1:  # Channel 2 - inverted
                channels.append(abs(-lsb_value))
            else:  # Other channels - variations with noise
                noise = random.randint(-100, 100)
                variation = int(lsb_value * (0.8 + 0.4 * random.random()))
                channels.append(abs(variation + noise))
        
        # Create metadata (3 bytes)
        # Byte 0: General flags
        pace_detected = random.random() < 0.01  # 1% chance
        saturation = random.random() < 0.001    # 0.1% chance
        flags = (int(saturation) << 1) | int(pace_detected)
        
        # Byte 1: Lead-off detection (random for simulation)
        lead_off_bits = random.randint(0, 255) if random.random() < 0.05 else 0
        
        # Byte 2: Sign bits (bit 7=Ch1, bit 0=Ch8)
        sign_bits = 0
        for i in range(8):
            if lsb_value < 0:  # If original signal is negative
                bit_pos = 7 - i  # Ch1=bit7, Ch8=bit0
                if i < 2:  # Apply to first 2 channels
                    sign_bits |= (1 << bit_pos)
        
        # Pack metadata
        metadata = struct.pack('BBB', flags, lead_off_bits, sign_bits)
        
        # Pack channel data (8 channels * 2 bytes each = 16 bytes)
        channel_data = struct.pack('>8H', *channels)  # Big-endian uint16
        
        return metadata + channel_data
    
    def _create_packet(self) -> bytes:
        """Create a complete packet with 50 samples."""
        # Packet header: 1B type + 4B number + 2B data_len
        packet_type = 1
        data_len = self.SAMPLES_PER_PACKET * self.SAMPLE_SIZE  # 50 * 19 = 950 bytes
        header = struct.pack('>BIH', packet_type, self.packet_number, data_len)
        
        # Create 50 samples
        samples_data = b''
        for i in range(self.SAMPLES_PER_PACKET):
            sample = self._create_sample(self.sample_index + i)
            samples_data += sample
        
        # Calculate CRC for header + data
        payload = header + samples_data
        crc = self._calculate_crc(payload)
        
        # Complete packet: STX + header + data + CRC + ETX
        packet = (
            struct.pack('>I', self.STX) +  # 4B STX
            payload +                       # 7B header + 950B data
            struct.pack('B', crc) +         # 1B CRC
            struct.pack('>I', self.ETX)     # 4B ETX
        )
        
        return packet
    
    def start_transmission(self):
        """Start transmitting ECG data."""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Serial port not connected!")
            return
        
        self.running = True
        print(f"Starting ECG transmission at {self.sample_rate} Hz sample rate, {self.packet_rate} Hz packet rate")
        print(f"Using waveform: {self.waveform.record_name}")
        print("Press Ctrl+C to stop...")
        
        packet_interval = 1.0 / self.packet_rate  # Time between packets
        samples_per_packet_time = self.SAMPLES_PER_PACKET / self.sample_rate
        
        try:
            while self.running:
                start_time = time.time()
                
                # Create and send packet
                packet = self._create_packet()
                self.serial_conn.write(packet)
                
                # Update counters
                self.packet_number = (self.packet_number + 1) % (2**32)
                self.sample_index += self.SAMPLES_PER_PACKET
                
                # Status update
                if self.packet_number % 100 == 0:
                    print(f"Sent packet {self.packet_number}, sample {self.sample_index}")
                
                # Wait for next packet time
                elapsed = time.time() - start_time
                sleep_time = max(0, packet_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nTransmission stopped by user")
        except Exception as e:
            print(f"Transmission error: {e}")
        finally:
            self.running = False
    
    def stop_transmission(self):
        """Stop transmitting data."""
        self.running = False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Mock ECG DAQ Hardware using AAMI EC13 waveforms")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM6, /dev/ttyUSB1)")
    parser.add_argument("--waveform", default="aami3a", help="AAMI waveform name (default: aami3a)")
    parser.add_argument("--waveform-path", default="test_data/aami-ec13", 
                       help="Path to AAMI waveform files")
    parser.add_argument("--sample-rate", type=float, default=500.0, 
                       help="ECG sample rate in Hz (default: 500)")
    parser.add_argument("--packet-rate", type=float, default=10.0,
                       help="Packet transmission rate in Hz (default: 10)")
    parser.add_argument("--list-waveforms", action="store_true",
                       help="List available waveforms and exit")
    
    args = parser.parse_args()
    
    # List available waveforms
    if args.list_waveforms:
        waveform_path = Path(args.waveform_path)
        if waveform_path.exists():
            hea_files = list(waveform_path.glob("*.hea"))
            print("Available AAMI EC13 waveforms:")
            for hea_file in hea_files:
                print(f"  {hea_file.stem}")
        else:
            print(f"Waveform path {waveform_path} does not exist")
        return
    
    # Create mock hardware
    mock_daq = MockDAQHardware(
        port=args.port,
        waveform_path=args.waveform_path,
        waveform_name=args.waveform,
        sample_rate=args.sample_rate,
        packet_rate=args.packet_rate
    )
    
    # Connect and start transmission
    if mock_daq.connect():
        try:
            mock_daq.start_transmission()
        finally:
            mock_daq.disconnect()
    else:
        print("Failed to connect to serial port")
        sys.exit(1)


if __name__ == "__main__":
    main()
