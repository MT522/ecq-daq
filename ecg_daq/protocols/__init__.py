"""Protocol implementations for ECG DAQ system."""

from .crc import CRC_FUNCTIONS, calculate_crc

__all__ = ["CRC_FUNCTIONS", "calculate_crc"]
