#!/usr/bin/env python3
"""
Test script for UART receiver using mock DAQ hardware.

This script sets up virtual serial ports and tests your UART receiver
with the mock DAQ hardware simulator.
"""

import asyncio
import threading
import time
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from examples.real_time_monitor import RealTimeMonitor
    from ecg_daq.core.config import Config, CRCMethod
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class VirtualSerialPortManager:
    """Manages virtual serial port creation for testing."""
    
    def __init__(self):
        self.port_pair = None
        self.process = None
    
    def create_virtual_ports(self):
        """Create virtual serial port pair (Windows/Linux compatible)."""
        import platform
        
        if platform.system() == "Windows":
            # On Windows, we'll use com0com or similar
            # For now, suggest using physical loopback or USB-to-serial adapters
            print("On Windows, please use:")
            print("1. Two USB-to-serial adapters with TX/RX crossed")
            print("2. com0com virtual serial port driver")
            print("3. Physical loopback adapter")
            return ("COM5", "COM6")  # Example ports
        else:
            # On Linux/Mac, use socat to create virtual serial ports
            try:
                # This would create /tmp/ttyV0 and /tmp/ttyV1
                cmd = ["socat", "-d", "-d", "pty,raw,echo=0", "pty,raw,echo=0"]
                self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
                time.sleep(1)  # Wait for ports to be created
                
                # Parse socat output to get port names
                # This is simplified - in practice you'd parse the actual output
                return ("/tmp/ttyV0", "/tmp/ttyV1")
            except Exception as e:
                print(f"Could not create virtual ports: {e}")
                return None
    
    def cleanup(self):
        """Clean up virtual ports."""
        if self.process:
            self.process.terminate()


def test_mock_hardware():
    """Test the mock hardware independently."""
    print("Testing mock DAQ hardware...")
    
    # Test waveform loading
    try:
        from mock_daq_hardware import WaveformReader
        
        waveform_path = "test_data/aami-ec13"
        if not Path(waveform_path).exists():
            print(f"Warning: {waveform_path} does not exist")
            print("Please download AAMI EC13 test files first")
            return False
        
        # Test loading a waveform
        reader = WaveformReader(waveform_path, "aami3a")
        
        # Test getting some samples
        for i in range(10):
            sample = reader.get_sample(i)
            print(f"Sample {i}: {sample:.6f} mV")
        
        print("Mock hardware test passed!")
        return True
        
    except Exception as e:
        print(f"Mock hardware test failed: {e}")
        return False


def run_mock_hardware(port: str, waveform: str = "aami3a"):
    """Run mock hardware in a separate thread."""
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "mock_daq_hardware.py",
        "--port", port,
        "--waveform", waveform,
        "--sample-rate", "500",
        "--packet-rate", "10"
    ]
    
    print(f"Starting mock hardware on {port}...")
    process = subprocess.Popen(cmd)
    return process


async def run_uart_receiver(port: str):
    """Run UART receiver."""
    print(f"Starting UART receiver on {port}...")
    
    try:
        # Create configuration
        config = Config.create_default(port)
        config.uart.baudrate = 115200
        config.ecg.sample_rate = 500.0
        config.protocol.crc_method = CRCMethod("sum")
        config.protocol.enable_crc = True
        
        # Create and start monitor
        monitor = RealTimeMonitor(config)
        await monitor.start()
        
    except KeyboardInterrupt:
        print("\nUART receiver stopped by user.")
    except Exception as e:
        print(f"UART receiver failed: {e}")
    finally:
        if 'monitor' in locals():
            await monitor.stop()


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test UART receiver with mock hardware")
    parser.add_argument("--mock-port", default="COM6", help="Port for mock hardware")
    parser.add_argument("--receiver-port", default="COM5", help="Port for UART receiver")
    parser.add_argument("--waveform", default="aami3a", help="AAMI waveform to use")
    parser.add_argument("--test-only", action="store_true", help="Test mock hardware only")
    parser.add_argument("--list-waveforms", action="store_true", help="List available waveforms")
    
    args = parser.parse_args()
    
    # List waveforms
    if args.list_waveforms:
        waveform_path = Path("test_data/aami-ec13")
        if waveform_path.exists():
            hea_files = list(waveform_path.glob("*.hea"))
            print("Available AAMI EC13 waveforms:")
            for hea_file in hea_files:
                print(f"  {hea_file.stem}")
        else:
            print("No waveform files found. Please download them first.")
        return
    
    # Test mock hardware
    if not test_mock_hardware():
        print("Mock hardware test failed!")
        return
    
    if args.test_only:
        print("Mock hardware test completed.")
        return
    
    print(f"\n=== UART Test Configuration ===")
    print(f"Mock Hardware Port: {args.mock_port}")
    print(f"UART Receiver Port: {args.receiver_port}")
    print(f"Waveform: {args.waveform}")
    print(f"===============================\n")
    
    print("Instructions:")
    print("1. Connect a serial cable between the two ports (TX->RX, RX->TX)")
    print("2. Or use virtual serial port software (com0com on Windows)")
    print("3. The mock hardware will transmit on one port")
    print("4. The UART receiver will listen on the other port")
    print("5. Press Ctrl+C to stop both")
    
    input("\nPress Enter to start the test...")
    
    # Start mock hardware
    mock_process = run_mock_hardware(args.mock_port, args.waveform)
    
    try:
        # Wait a moment for mock hardware to start
        time.sleep(2)
        
        # Start UART receiver
        asyncio.run(run_uart_receiver(args.receiver_port))
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        # Clean up mock hardware
        if mock_process:
            mock_process.terminate()
            mock_process.wait()
        print("Test completed.")


if __name__ == "__main__":
    main()
