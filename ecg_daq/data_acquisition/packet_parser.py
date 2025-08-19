"""Packet parser for ECG DAQ system."""

import struct
from typing import List, Optional
import numpy as np

from ..core.config import Config
from ..core.models import Packet, Sample
from ..core.exceptions import PacketParseError, CRCError
from ..protocols.crc import validate_crc


class PacketParser:
    """High-performance packet parser for ECG data."""
    
    def __init__(self, config: Config):
        """
        Initialize packet parser.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._sample_size = config.protocol.sample_size
        self._channel_count = config.protocol.channel_count
        self._samples_per_packet = config.protocol.samples_per_packet
        
        # Pre-compile format strings for performance
        self._header_format = config.protocol.header_format
        self._channel_format = self._get_channel_format()
        self._header_len = self._calculate_header_length()
        
        # CRC settings
        self._crc_method = config.protocol.crc_method.value
        self._enable_crc = config.protocol.enable_crc
    
    def _calculate_header_length(self) -> int:
        """Calculate header length from format string."""
        # Parse format string to determine header size
        # Format: ">B I H" -> 1 + 4 + 2 = 7 bytes
        size_map = {'B': 1, 'H': 2, 'I': 4, 'L': 4, 'Q': 8}
        total_size = 0
        
        for char in self._header_format:
            if char in size_map:
                total_size += size_map[char]
        
        return total_size
    
    def _get_channel_format(self) -> str:
        """Get format string for channel data."""
        endian = ">" if self.config.protocol.endianness.value == "big" else "<"
        return f"{endian}{self._channel_count}H"
    
    def parse_packet(self, data: bytes) -> Packet:
        """
        Parse a complete packet from raw data.
        
        Args:
            data: Raw packet data (excluding start/stop words)
            
        Returns:
            Parsed Packet object
            
        Raises:
            PacketParseError: If packet parsing fails
        """
        try:
            # Validate packet length
            if len(data) < self._header_len + 1:  # +1 for CRC
                raise PacketParseError(f"Packet too short: {len(data)} < {self._header_len + 1}")
            
            # Parse header
            header = data[:self._header_len]
            header_values = struct.unpack(self._header_format, header)
            
            if len(header_values) < 3:
                raise PacketParseError(f"Invalid header format: expected 3 values, got {len(header_values)}")
            
            packet_type, packet_number, data_size = header_values
            
            # Validate packet length
            # Expected: header (7B) + data (19B * 50 samples) + CRC (1B) = total
            expected_data_size = self._sample_size * self._samples_per_packet  # 19 * 50 = 950 bytes
            expected_len = self._header_len + expected_data_size + self._get_crc_size()
            
            if data_size != expected_data_size:
                raise PacketParseError(
                    f"Data size mismatch: header says {data_size}, expected {expected_data_size}"
                )
            
            if len(data) != expected_len:
                raise PacketParseError(
                    f"Packet length mismatch: got {len(data)}, expected {expected_len}"
                )
            
            # Extract data section
            data_section = data[self._header_len:self._header_len + data_size]
            
            # Validate CRC if enabled
            if self._enable_crc:
                crc_section = data[self._header_len + data_size:]
                if not self._validate_crc(data[:self._header_len + data_size], crc_section):
                    raise CRCError("CRC validation failed")
            
            # Parse samples
            samples = self._parse_samples(data_section)
            
            # Create packet
            return Packet(
                packet_type=packet_type,
                packet_number=packet_number,
                samples=samples,
                raw_data_size=data_size
            )
            
        except struct.error as e:
            raise PacketParseError(f"Header parsing failed: {e}")
        except Exception as e:
            if not isinstance(e, (PacketParseError, CRCError)):
                raise PacketParseError(f"Packet parsing failed: {e}")
            raise
    
    def _parse_samples(self, data: bytes) -> List[Sample]:
        """
        Parse sample data from data section.
        
        Args:
            data: Raw sample data (950 bytes = 50 samples * 19 bytes each)
            
        Returns:
            List of Sample objects (50 samples)
        """
        expected_total_size = self._sample_size * self._samples_per_packet
        if len(data) != expected_total_size:
            raise PacketParseError(
                f"Data length {len(data)} != expected {expected_total_size} "
                f"({self._samples_per_packet} samples * {self._sample_size} bytes)"
            )
        
        samples = []
        
        for i in range(self._samples_per_packet):  # Process exactly 50 samples
            sample_start = i * self._sample_size
            sample_end = sample_start + self._sample_size
            sample_data = data[sample_start:sample_end]
            
            sample = self._parse_single_sample(sample_data)
            samples.append(sample)
        
        return samples
    
    def _parse_single_sample(self, sample_data: bytes) -> Sample:
        """
        Parse a single sample from raw data.
        
        Args:
            sample_data: Raw sample data (19 bytes: 3B metadata + 16B channels)
            
        Returns:
            Sample object
        """
        if len(sample_data) != self._sample_size:
            raise PacketParseError(
                f"Sample data length {len(sample_data)} != expected {self._sample_size}"
            )
        
        # Parse metadata (first 3 bytes)
        metadata = sample_data[:3]
        b0, b1, b2 = metadata
        
        # Extract flags from metadata
        # b0: General flags (you can define these based on your protocol)
        saturation = bool((b0 >> 1) & 0x01)
        pace_detected = bool(b0 & 0x01)
        
        # b1: Lead-off detection bits 
        # 0b: Not detected, 1b: Detected
        lead_off_bits = b1
        
        # b2: Sign bits (bit 7 = Channel 1, bit 6 = Channel 2, ..., bit 0 = Channel 8) 
        # 0b: Positive, 1b: Negative
        sign_bits = b2
        
        # Parse channel data (16 bytes = 8 channels * 2 bytes each)
        channel_data = sample_data[3:]
        if len(channel_data) != 16:  # Must be exactly 16 bytes
            raise PacketParseError(
                f"Channel data length {len(channel_data)} != expected 16 bytes"
            )
        
        # Unpack 8 channels as uint16 values
        try:
            channel_values = struct.unpack(self._channel_format, channel_data)
        except struct.error as e:
            raise PacketParseError(f"Channel data unpacking failed: {e}")
        
        # Apply sign bits and convert to millivolts with high precision
        # Each LSB = 10 µV, convert to mV: LSB * 10µV * (1mV/1000µV) = LSB * 0.01 mV
        LSB_TO_MV = 0.01  # 10 µV per LSB converted to mV
        
        signed_channels = []
        for i, value in enumerate(channel_values):
            # Apply sign bit for channel (i+1)
            # Bit mapping: bit 7 = Channel 1, bit 6 = Channel 2, ..., bit 0 = Channel 8
            # 0b: Positive, 1b: Negative
            bit_position = 7 - i  # Channel 1 uses bit 7, Channel 8 uses bit 0
            is_negative = bool((sign_bits >> bit_position) & 0x01)
            signed_value = -value if is_negative else value
            
            # Convert to millivolts with high precision
            mv_value = signed_value * LSB_TO_MV
            signed_channels.append(mv_value)
        
        # Create Sample object
        return Sample(
            channels=signed_channels,
            saturation=saturation,
            pace_detected=pace_detected,
            lead_off_bits=lead_off_bits,
            sign_bits=sign_bits,
            raw_metadata=metadata
        )
    
    def _validate_crc(self, payload: bytes, crc_data: bytes) -> bool:
        """
        Validate CRC for payload.
        
        Args:
            payload: Data to validate
            crc_data: Received CRC data
            
        Returns:
            True if CRC is valid
        """
        if not crc_data:
            return False
        
        # Extract CRC value based on method
        crc_size = self._get_crc_size()
        if len(crc_data) < crc_size:
            return False
        
        # Convert CRC bytes to integer
        if crc_size == 1:
            received_crc = crc_data[0]
        elif crc_size == 2:
            received_crc = struct.unpack(">H", crc_data[:2])[0]
        elif crc_size == 4:
            received_crc = struct.unpack(">I", crc_data[:4])[0]
        else:
            return False
        
        # Validate CRC
        return validate_crc(payload, received_crc, self._crc_method)
    
    def _get_crc_size(self) -> int:
        """Get CRC size in bytes for current method."""
        if self._crc_method.startswith('crc16'):
            return 2
        elif self._crc_method.startswith('crc32'):
            return 4
        else:
            return 1
    
    def create_sample_from_numpy(self, data: np.ndarray, **kwargs) -> Sample:
        """
        Create a Sample object from numpy array.
        
        Args:
            data: Channel data as numpy array
            **kwargs: Additional sample attributes
            
        Returns:
            Sample object
        """
        if data.size > self._channel_count:
            raise ValueError(f"Data has {data.size} channels, maximum is {self._channel_count}")
        
        # Pad with zeros if needed
        channels = list(data.astype(np.float32))
        while len(channels) < self._channel_count:
            channels.append(0.0)
        
        return Sample(channels=channels, **kwargs)
    
    def get_packet_info(self, data: bytes) -> dict:
        """
        Get basic packet information without full parsing.
        
        Args:
            data: Raw packet data
            
        Returns:
            Dictionary with packet info
        """
        try:
            if len(data) < self._header_len:
                return {"error": "Data too short for header"}
            
            header = data[:self._header_len]
            header_values = struct.unpack(self._header_format, header)
            
            return {
                "packet_type": header_values[0] if len(header_values) > 0 else None,
                "packet_number": header_values[1] if len(header_values) > 1 else None,
                "data_size": header_values[2] if len(header_values) > 2 else None,
                "total_length": len(data),
                "header_length": self._header_len,
                "crc_size": self._get_crc_size()
            }
        except Exception as e:
            return {"error": str(e)}
