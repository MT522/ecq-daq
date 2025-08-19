"""Custom exceptions for ECG DAQ system."""


class ECGDAQError(Exception):
    """Base exception for ECG DAQ system."""
    pass


class ConfigurationError(ECGDAQError):
    """Raised when there's a configuration error."""
    pass


class ProtocolError(ECGDAQError):
    """Raised when there's a protocol violation."""
    pass


class DataAcquisitionError(ECGDAQError):
    """Raised when there's an error in data acquisition."""
    pass


class VisualizationError(ECGDAQError):
    """Raised when there's an error in visualization."""
    pass


class CRCError(ProtocolError):
    """Raised when CRC validation fails."""
    pass


class PacketParseError(ProtocolError):
    """Raised when packet parsing fails."""
    pass
