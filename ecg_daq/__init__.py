"""ECG DAQ - High-performance ECG data acquisition system."""

__version__ = "0.1.0"
__author__ = "Mehrshad"
__email__ = "mehrshad@example.com"

# Core imports for easy access
from .core.config import Config, ECGConfig, UARTConfig, ProtocolConfig
from .core.models import Packet, Sample, ECGData, SystemStatus
from .core.exceptions import ECGDAQError, PacketParseError, CRCError, ConfigError
from .data_acquisition.uart_receiver import AsyncUARTReceiver
from .data_acquisition.packet_parser import PacketParser
from .protocols.crc import CRCMethod, validate_crc

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Configuration
    "Config",
    "ECGConfig", 
    "UARTConfig",
    "ProtocolConfig",
    
    # Data models
    "Packet",
    "Sample",
    "ECGData",
    "SystemStatus",
    
    # Exceptions
    "ECGDAQError",
    "PacketParseError",
    "CRCError",
    "ConfigError",
    
    # Core components
    "AsyncUARTReceiver",
    "PacketParser",
    
    # Protocol utilities
    "CRCMethod",
    "validate_crc",
]
