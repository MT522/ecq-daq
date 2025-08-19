"""Data models for ECG DAQ system."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
import numpy as np


class Sample(BaseModel):
    """A single ECG sample with metadata."""
    channels: List[float] = Field(..., description="Channel values (μV)")
    saturation: bool = Field(default=False, description="ADC saturation flag")
    pace_detected: bool = Field(default=False, description="Pace detection flag")
    lead_off_bits: int = Field(default=0, ge=0, le=255, description="Lead-off detection bits")
    sign_bits: int = Field(default=0, ge=0, le=255, description="Sign bits for channels")
    raw_metadata: bytes = Field(default=b'\x00\x00\x00', description="Raw metadata bytes")
    timestamp: Optional[float] = Field(default=None, description="Sample timestamp")
    
    @validator('channels')
    def validate_channels(cls, v):
        """Validate channel data."""
        if not v:
            raise ValueError("Channels cannot be empty")
        if len(v) > 12:
            raise ValueError("Maximum 12 channels supported")
        return v
    
    @property
    def channel_count(self) -> int:
        """Get number of channels."""
        return len(self.channels)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.channels, dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, data: np.ndarray, **kwargs) -> 'Sample':
        """Create from numpy array."""
        return cls(channels=data.tolist(), **kwargs)


class Packet(BaseModel):
    """A complete data packet."""
    packet_type: int = Field(..., ge=0, le=255, description="Packet type identifier")
    packet_number: int = Field(..., ge=0, description="Sequential packet number")
    samples: List[Sample] = Field(..., description="List of samples in packet")
    raw_data_size: int = Field(..., ge=0, description="Raw data size in bytes")

    timestamp: Optional[float] = Field(default=None, description="Packet timestamp")
    
    @validator('samples')
    def validate_samples(cls, v):
        """Validate samples."""
        if not v:
            raise ValueError("Packet must contain at least one sample")
        return v
    
    @property
    def sample_count(self) -> int:
        """Get number of samples in packet."""
        return len(self.samples)
    
    @property
    def total_channels(self) -> int:
        """Get total number of channels across all samples."""
        return sum(sample.channel_count for sample in self.samples)


class ECGData(BaseModel):
    """ECG data container for visualization."""
    samples: List[Sample] = Field(..., description="List of ECG samples")
    start_time: float = Field(..., description="Start time of data")
    end_time: float = Field(..., description="End time of data")
    sample_rate: float = Field(..., gt=0, description="Sample rate in Hz")
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        """Validate time range."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError("End time must be greater than start time")
        return v
    
    @property
    def duration(self) -> float:
        """Get data duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.samples)
    
    def get_channel_data(self, channel_idx: int) -> np.ndarray:
        """Get data for a specific channel."""
        if channel_idx < 0 or channel_idx >= 12:
            raise ValueError("Channel index must be between 0 and 11")
        
        data = []
        for sample in self.samples:
            if channel_idx < len(sample.channels):
                data.append(sample.channels[channel_idx])
            else:
                data.append(0.0)  # Pad with zeros if channel doesn't exist
        
        return np.array(data, dtype=np.float32)
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis for plotting."""
        return np.linspace(self.start_time, self.end_time, self.sample_count)
    
    def to_numpy(self) -> np.ndarray:
        """Convert all channel data to numpy array."""
        # Create a 2D array: channels x samples
        max_channels = max(sample.channel_count for sample in self.samples)
        data = np.zeros((max_channels, self.sample_count), dtype=np.float32)
        
        for i, sample in enumerate(self.samples):
            for j, value in enumerate(sample.channels):
                if j < max_channels:
                    data[j, i] = value
        
        return data


class SystemStatus(BaseModel):
    """System status information."""
    is_running: bool = Field(default=False, description="System running status")
    packets_received: int = Field(default=0, description="Total packets received")
    samples_processed: int = Field(default=0, description="Total samples processed")
    errors_count: int = Field(default=0, description="Total error count")
    uptime: float = Field(default=0.0, description="System uptime in seconds")
    current_sample_rate: float = Field(default=0.0, description="Current sample rate")
    buffer_utilization: float = Field(default=0.0, ge=0.0, le=1.0, description="Buffer utilization")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


# Legacy dataclass for backward compatibility
@dataclass
class LegacySample:
    """Legacy sample class for backward compatibility."""
    channels: List[int]
    saturation: bool
    pace_detected: bool
    lead_off_bits: int
    sign_bits: int
    raw_metadata: bytes
    
    def to_sample(self) -> Sample:
        """Convert to new Sample model."""
        return Sample(
            channels=[float(ch) for ch in self.channels],
            saturation=self.saturation,
            pace_detected=self.pace_detected,
            lead_off_bits=self.lead_off_bits,
            sign_bits=self.sign_bits,
            raw_metadata=self.raw_metadata
        )
