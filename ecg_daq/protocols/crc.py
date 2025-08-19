"""CRC implementations for ECG DAQ system."""

from typing import Dict, Callable, Union
import numpy as np
from ..core.exceptions import CRCError


def crc8_sum(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-8 Sum (simple byte sum modulo 256)."""
    if isinstance(payload, np.ndarray):
        return int(np.sum(payload) & 0xFF)
    return sum(payload) & 0xFF


def crc8_maxim(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-8/MAXIM (polynomial 0x31, initial value 0x00)."""
    crc = 0x00
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x31
            else:
                crc <<= 1
        crc &= 0xFF
    return crc


def crc8_ccitt(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-8/CCITT (polynomial 0x07, initial value 0x00)."""
    crc = 0x00
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
        crc &= 0xFF
    return crc


def crc8_dallas(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-8/Dallas (polynomial 0x31, initial value 0x00, final XOR 0x00)."""
    crc = 0x00
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8C
            else:
                crc >>= 1
    return crc


def crc8_sae_j1850(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-8/SAE J1850 (polynomial 0x1D, initial value 0xFF, final XOR 0xFF)."""
    crc = 0xFF
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x1D
            else:
                crc <<= 1
        crc &= 0xFF
    return crc ^ 0xFF


def crc8_rohc(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-8/ROHC (polynomial 0x07, initial value 0xFF, final XOR 0x00)."""
    crc = 0xFF
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
        crc &= 0xFF
    return crc


def crc16_ccitt(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-16/CCITT (polynomial 0x1021, initial value 0xFFFF)."""
    crc = 0xFFFF
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
        crc &= 0xFFFF
    return crc


def crc16_modbus(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-16/MODBUS (polynomial 0x8005, initial value 0xFFFF)."""
    crc = 0xFFFF
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def crc16_xmodem(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-16/XMODEM (polynomial 0x1021, initial value 0x0000)."""
    crc = 0x0000
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
        crc &= 0xFFFF
    return crc


def crc32_ieee(payload: Union[bytes, np.ndarray]) -> int:
    """CRC-32/IEEE (polynomial 0x04C11DB7, initial value 0xFFFFFFFF, final XOR 0xFFFFFFFF)."""
    crc = 0xFFFFFFFF
    if isinstance(payload, np.ndarray):
        payload = payload.tobytes()
    
    for byte in payload:
        crc ^= (byte << 24)
        for _ in range(8):
            if crc & 0x80000000:
                crc = (crc << 1) ^ 0x04C11DB7
            else:
                crc <<= 1
        crc &= 0xFFFFFFFF
    return crc ^ 0xFFFFFFFF


# Dictionary of available CRC functions
CRC_FUNCTIONS: Dict[str, Callable[[Union[bytes, np.ndarray]], int]] = {
    'sum': crc8_sum,
    'maxim': crc8_maxim,
    'ccitt': crc8_ccitt,
    'dallas': crc8_dallas,
    'sae_j1850': crc8_sae_j1850,
    'rohc': crc8_rohc,
    'crc16_ccitt': crc16_ccitt,
    'crc16_modbus': crc16_modbus,
    'crc16_xmodem': crc16_xmodem,
    'crc32': crc32_ieee
}


def calculate_crc(payload: Union[bytes, np.ndarray], method: str = 'sum') -> int:
    """
    Calculate CRC for payload using specified method.
    
    Args:
        payload: Data to calculate CRC for
        method: CRC method to use
        
    Returns:
        Calculated CRC value
        
    Raises:
        CRCError: If method is not supported
    """
    if method not in CRC_FUNCTIONS:
        raise CRCError(f"Unsupported CRC method: {method}")
    
    try:
        return CRC_FUNCTIONS[method](payload)
    except Exception as e:
        raise CRCError(f"CRC calculation failed: {e}")


def validate_crc(payload: Union[bytes, np.ndarray], received_crc: int, method: str = 'sum') -> bool:
    """
    Validate CRC for payload.
    
    Args:
        payload: Data to validate
        received_crc: Received CRC value
        method: CRC method to use
        
    Returns:
        True if CRC is valid, False otherwise
    """
    try:
        calculated_crc = calculate_crc(payload, method)
        return calculated_crc == received_crc
    except CRCError:
        return False


def get_crc_size(method: str) -> int:
    """
    Get CRC size in bytes for specified method.
    
    Args:
        method: CRC method name
        
    Returns:
        CRC size in bytes
    """
    if method.startswith('crc16'):
        return 2
    elif method.startswith('crc32'):
        return 4
    else:
        return 1


def list_available_methods() -> list:
    """Get list of available CRC methods."""
    return list(CRC_FUNCTIONS.keys())
