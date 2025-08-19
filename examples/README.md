# ECG DAQ Examples

This directory contains example scripts and usage demonstrations for the ECG DAQ system.

## Available Examples

### `real_time_monitor.py`
Complete real-time ECG monitoring system with data logging capabilities.

**Features:**
- Asynchronous UART communication
- High-performance packet parsing
- CSV data logging with microsecond timestamps
- Missing packet detection and recovery
- Lead-off and sign bit interpretation
- Comprehensive status monitoring

**Usage:**
```bash
python real_time_monitor.py --port COM5 --baudrate 115200 --sample-rate 500
```

**Configuration:**
- Port: Serial port (COM5, /dev/ttyUSB0, etc.)
- Baud rate: 9600 to 921600 bps
- Sample rate: 100 to 10000 Hz
- CRC method: sum, crc16, crc32

## Running Examples

### Prerequisites
```bash
pip install -e ".[dev]"
```

### Basic Monitoring
```bash
# Windows
python examples/real_time_monitor.py --port COM5

# Linux
python examples/real_time_monitor.py --port /dev/ttyUSB0

# macOS
python examples/real_time_monitor.py --port /dev/cu.usbserial-1420
```

### Advanced Configuration
```bash
python examples/real_time_monitor.py \
    --port COM5 \
    --baudrate 115200 \
    --sample-rate 1000 \
    --crc-method crc16 \
    --output-dir data/
```

## Output Files

The monitoring system creates timestamped CSV files:
```
data/
├── ecg_data_20231215_143022.csv
├── ecg_data_20231215_150145.csv
└── ...
```

## Integration Examples

### Using as Library
```python
import asyncio
from ecg_daq import Config, AsyncUARTReceiver
from examples.real_time_monitor import RealTimeMonitor

async def custom_monitoring():
    config = Config.create_default("COM5")
    config.ecg.sample_rate = 500.0
    
    monitor = RealTimeMonitor(config)
    await monitor.start()

asyncio.run(custom_monitoring())
```

### Custom Packet Handling
```python
from ecg_daq import AsyncUARTReceiver, PacketParser

def custom_packet_handler(packet):
    print(f"Received packet {packet.packet_number}")
    for i, sample in enumerate(packet.samples):
        print(f"Sample {i}: {sample.channels}")

# Setup receiver with custom handler
receiver = AsyncUARTReceiver(config)
receiver.set_packet_callback(custom_packet_handler)
```

## Troubleshooting

### Common Issues
- **Permission denied**: Add user to dialout group on Linux
- **Port not found**: Check device manager or use `ls /dev/tty*`
- **No data received**: Verify baud rate and hardware connections
- **High CPU usage**: Reduce sample rate or increase packet size

### Debug Mode
```bash
python examples/real_time_monitor.py --port COM5 --debug
```

This enables detailed logging for troubleshooting connection and parsing issues.
