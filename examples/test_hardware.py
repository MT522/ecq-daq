#!/usr/bin/env python3
"""Test script to verify hardware connection and data format."""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ecgdaq.core.config import Config
from ecgdaq.data_acquisition.uart_receiver import AsyncUARTReceiver


async def test_hardware_connection(port: str, baudrate: int = 115200, duration: int = 30):
    """
    Test hardware connection and display raw data.
    
    Args:
        port: Serial port to test
        baudrate: Baud rate
        duration: Test duration in seconds
    """
    print(f"Testing hardware connection on {port} @ {baudrate} baud")
    print(f"Test duration: {duration} seconds")
    print("-" * 50)
    
    # Create configuration
    config = Config.create_default(port)
    config.uart.baudrate = baudrate
    config.protocol.enable_crc = False  # Disable CRC for testing
    
    # Create receiver
    receiver = AsyncUARTReceiver(config)
    
    try:
        # Start receiver
        await receiver.start()
        print("UART receiver started successfully.")
        
        # Test data reception
        start_time = time.time()
        packet_count = 0
        sample_count = 0
        
        print("Receiving data... (Press Ctrl+C to stop early)")
        
        async for packet in receiver.packets():
            packet_count += 1
            sample_count += len(packet.samples)
            
            # Print packet info
            print(f"\nPacket #{packet.packet_number}:")
            print(f"  Type: 0x{packet.packet_type:02X}")
            print(f"  Data Size: {packet.raw_data_size} bytes")
            print(f"  Samples: {len(packet.samples)}")
            
            # Print first sample details
            if packet.samples:
                sample = packet.samples[0]
                print(f"  First Sample:")
                print(f"    Channels: {sample.channels[:4]}...")  # Show first 4 channels
                print(f"    Saturation: {sample.saturation}")
                print(f"    Pace Detected: {sample.pace_detected}")
                print(f"    Lead-off Bits: 0b{sample.lead_off_bits:08b}")
                print(f"    Sign Bits: 0b{sample.sign_bits:08b}")
            
            # Check if we've run long enough
            if time.time() - start_time >= duration:
                break
            
            # Limit output to avoid spam
            if packet_count >= 10:
                print(f"\n... received {packet_count} packets, {sample_count} samples total")
                print("Continuing to receive data... (Press Ctrl+C to stop)")
                break
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\n" + "="*50)
        print(f"TEST COMPLETED")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Packets Received: {packet_count}")
        print(f"Total Samples: {sample_count}")
        if elapsed > 0:
            print(f"Average Sample Rate: {sample_count/elapsed:.1f} Hz")
            print(f"Average Packet Rate: {packet_count/elapsed:.1f} packets/sec")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Test failed: {e}")
        raise
    finally:
        await receiver.stop()
        print("UART receiver stopped.")


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Connection Test")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM5, /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    try:
        await test_hardware_connection(args.port, args.baudrate, args.duration)
    except Exception as e:
        print(f"Hardware test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
