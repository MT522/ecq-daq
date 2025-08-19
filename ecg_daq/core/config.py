"""Configuration management for ECG DAQ system."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class CRCMethod(str, Enum):
    """Available CRC methods."""
    SUM = "sum"
    MAXIM = "maxim"
    CCITT = "ccitt"
    DALLAS = "dallas"
    SAE_J1850 = "sae_j1850"
    ROHC = "rohc"
    CRC16_CCITT = "crc16_ccitt"
    CRC16_MODBUS = "crc16_modbus"
    CRC16_XMODEM = "crc16_xmodem"
    CRC32 = "crc32"


class Endianness(str, Enum):
    """Byte endianness options."""
    BIG = "big"
    LITTLE = "little"


class UARTConfig(BaseModel):
    """UART communication configuration."""
    port: str = Field(..., description="Serial port (e.g., COM5, /dev/ttyUSB0)")
    baudrate: int = Field(default=115200, ge=9600, le=4000000, description="Baud rate")
    timeout: float = Field(default=0.1, gt=0, description="Read timeout in seconds")
    bytesize: int = Field(default=8, ge=5, le=8, description="Data bits")
    parity: str = Field(default="N", pattern="^[NOEMS]$", description="Parity")
    stopbits: float = Field(default=1.0, ge=1.0, le=2.0, description="Stop bits")


class ECGConfig(BaseModel):
    """ECG data acquisition configuration."""
    sample_rate: float = Field(default=500.0, ge=100.0, le=10000.0, description="Sample rate in Hz")


class ProtocolConfig(BaseModel):
    """Protocol configuration."""
    start_word: int = Field(default=0x11223344, description="Start word (4 bytes)")
    stop_word: int = Field(default=0xAABBCCDD, description="Stop word (4 bytes)")
    header_format: str = Field(default=">B I H", description="Header format: 1B packet_type + 4B packet_num + 2B data_len")
    sample_size: int = Field(default=19, description="Sample size in bytes (3B metadata + 16B channels)")
    samples_per_packet: int = Field(default=50, description="Number of samples per packet")
    channel_count: int = Field(default=8, description="Number of channels")
    endianness: Endianness = Field(default=Endianness.BIG, description="Channel word endianness")
    crc_method: CRCMethod = Field(default=CRCMethod.SUM, description="CRC method to use")
    enable_crc: bool = Field(default=True, description="Enable CRC validation")


class Config(BaseModel):
    """Main configuration for ECG DAQ system."""
    uart: UARTConfig
    ecg: ECGConfig = Field(default_factory=ECGConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    
    # Advanced options
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    enable_logging: bool = Field(default=True)
    max_buffer_size: int = Field(default=1_000_000, description="Maximum buffer size in bytes")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
    
    @validator('uart')
    def validate_uart_config(cls, v):
        """Validate UART configuration."""
        if not v.port:
            raise ValueError("UART port must be specified")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(**data)
    
    @classmethod
    def create_default(cls, port: str) -> 'Config':
        """Create default configuration for a given port."""
        return cls(
            uart=UARTConfig(port=port),
            ecg=ECGConfig(),
            protocol=ProtocolConfig()
        )
