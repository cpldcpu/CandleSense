# Zero Parts Candle Flicker Sensor

Detecting candle flame flicker using only a few wires and a CH32V003 microcontroller. 


<p align="center">
  <img src="media/flameosc_small.gif" alt="Device overview"/>
</p>

## Overview

This project uses the ADC in the MCU to implement a capacitive sensor that is able to detect motion of a candle flame and can extract the flicker frequency. The system samples the capacitive sensor, applies filtering and signal processing, and outputs processed data for analysis via the CH32Fun framework.

In addition, the flicker frequency is used as a frequency normal to blink a status LED at 1 Hz

### Pin Configuration

- **PA2**: Capacitive sensor input (sensed vs GND)
- **PC4**: Status LED output

## Project Structure

```
├── src/
│   ├── main.c              # Main firmware code
│   ├── funconfig.h         # CH32Fun configuration
│   └── ch32fun/            # CH32Fun framework (git submodule)
├── analyze_data.py         # Offline data analysis 
├── plot_realtime_qt.py     # Real-time data visualization and collection
├── Makefile               # Build configuration
└── *.csv                  # Collected sensor data files
```

## Firmware Operation

1. **Capacitive Sensing**: Uses CH32V003's ADC-based capacitive touch sensing from the CH32Fun framework.
2. **Signal Processing**:
   - Low-pass filter: `avg = avg - (avg>>5) + raw_value`
   - High-pass filter: `hp = raw_value - (avg>>5)`
3. **Zero Crossing Detection**: Detects positive transitions in the high-pass signal
5. **LED Control**: Uses a residual counter to convert 9.9 Hz flicker frequency to 1 Hz LED blink rate

### Data Output Format

The firmware outputs tab-separated values:
```
raw    avg    hp    zero_cross
46747  1495604  10   0
46734  1495601  -3   0
```

Where:
- `raw`: Raw ADC reading
- `avg`: Low-pass filtered average (×32 scaling)
- `hp`: High-pass filtered signal (flame detection)
- `zero_cross`: Binary flag for zero crossing events

## Software Tools

### Real-time Monitoring (`plot_realtime_qt.py`)

Real-time plotter using PyQtGraph.
- **Three-panel display**: Raw/filtered signals, flicker detection, zero crossings
- **Data logging**: Automatic CSV file creation with timestamps

**Dependencies:**
```bash
pip install pyqtgraph PyQt5 numpy
```

**Usage:**
```bash
sudo make monitor | python3 plot_realtime_qt.py
```

### Data Analysis (`analyze_data.py`)

Will read and analyze collected data files, showing time series plots of raw data, touch detection, and frequency analysis.

**Dependencies:**
```bash
pip install pandas matplotlib numpy scipy
```

**Usage:**
```bash
python3 analyze_data.py <csv_file>
```

## Building and Flashing

### Initial Setup

```bash
# Clone with submodules
git clone --recursive <repository-url>
```

### Build Commands
```bash
# Build and flash firmware
make flash

# Monitor output (requires appropriate permissions)
make monitor
```
```bash
# Build and flash firmware
make flash

# Monitor output (requires appropriate permissions)
make monitor
```

