"""ECG DAQ Examples module."""

from .real_time_monitor import RealTimeMonitor, main as monitor_main

# Import main functions from other examples
try:
    from .plot_ecg_data import main as plot_main
except ImportError:
    plot_main = None

try:
    from .mock_daq_hardware import main as mock_main
except ImportError:
    mock_main = None

try:
    from .test_uart_with_mock import main as test_main
except ImportError:
    test_main = None

__all__ = [
    "RealTimeMonitor", 
    "monitor_main", 
    "plot_main", 
    "mock_main", 
    "test_main"
]
