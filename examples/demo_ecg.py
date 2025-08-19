#!/usr/bin/env python3
"""Demo ECG plotting with synthetic data using async architecture."""

import asyncio
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ecgdaq.core.config import Config, PlottingMode
from ecgdaq.core.models import Sample
from ecgdaq.visualization.ecg_plotter import ECGPlotter


async def generate_synthetic_ecg(plotter: ECGPlotter, duration: float = 30.0):
    """
    Generate synthetic ECG data asynchronously.
    
    Args:
        plotter: ECG plotter instance
        duration: Duration to run in seconds
    """
    print(f"Generating synthetic ECG data for {duration} seconds...")
    
    start_time = time.time()
    sample_rate = plotter.sample_rate
    sample_interval = 1.0 / sample_rate
    
    t = 0
    while (time.time() - start_time) < duration and plotter.running:
        # Generate synthetic ECG waveform
        time_val = t * sample_interval
        
        # Create realistic ECG pattern
        ecg_pattern = create_ecg_waveform(time_val)
        
        # Create sample with 8 channels
        sample = Sample(
            channels=[ecg_pattern + i*100 for i in range(8)],
            saturation=False,
            pace_detected=False,
            lead_off_bits=0,
            sign_bits=0,
            raw_metadata=b'\x00\x00\x00'
        )
        
        # Update plotter
        plotter.update_data([sample])
        
        t += 1
        await asyncio.sleep(sample_interval)
    
    print("Synthetic data generation completed.")


def create_ecg_waveform(t: float) -> float:
    """
    Create a realistic ECG waveform.
    
    Args:
        t: Time value
        
    Returns:
        ECG amplitude
    """
    # Heart rate: 60 BPM = 1 Hz
    heart_rate = 1.0
    cycle_time = t % (1.0 / heart_rate)
    
    # P wave (atrial depolarization)
    p_wave = 100 * np.exp(-((cycle_time - 0.1) / 0.02)**2)
    
    # QRS complex (ventricular depolarization)
    qrs_complex = 800 * np.exp(-((cycle_time - 0.5) / 0.05)**2)
    
    # T wave (ventricular repolarization)
    t_wave = 200 * np.exp(-((cycle_time - 0.7) / 0.08)**2)
    
    # Combine waves
    ecg = p_wave + qrs_complex + t_wave
    
    # Add some noise for realism
    noise = np.random.normal(0, 15)
    ecg += noise
    
    return ecg


async def main():
    """Main demo function."""
    print("ECG DAQ Demo - Synthetic Data")
    print("=" * 40)
    
    # Create configuration
    config = Config.create_default("DEMO")
    config.ecg.plotting_mode = PlottingMode.BLOCKING
    config.ecg.buffer_size = 2000
    config.ecg.sample_rate = 500.0
    
    # Create and start plotter
    plotter = ECGPlotter(
        buffer_size=config.ecg.buffer_size,
        sample_rate=config.ecg.sample_rate
    )
    
    try:
        # Start the plotter
        plotter.start_animation()
        
        # Start synthetic data generation
        data_task = asyncio.create_task(
            generate_synthetic_ecg(plotter, duration=30.0)
        )
        
        # Show the plot
        print("Starting ECG plotter...")
        plotter.show_plot()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        # Cleanup
        plotter.stop_animation()
        if not data_task.done():
            data_task.cancel()
            try:
                await data_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)
