#!/usr/bin/env python3
"""
ECG Data Plotter - Standalone utility to plot ECG data from CSV files
Created by ECG DAQ system data logger.

Usage:
    python plot_ecg_data.py ecg_data_20231215_143022.csv
    python plot_ecg_data.py ecg_data_20231215_143022.csv --channels 1,2,3
    python plot_ecg_data.py ecg_data_20231215_143022.csv --time-window 10
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import sys
from pathlib import Path


class ECGDataPlotter:
    """Standalone ECG data plotter for CSV files."""
    
    def __init__(self, csv_file: str):
        """
        Initialize the plotter with a CSV file.
        
        Args:
            csv_file: Path to the CSV file containing ECG data
        """
        self.csv_file = Path(csv_file)
        self.data = None
        self.sample_rate = 500.0  # Default sample rate
        
    def load_data(self):
        """Load ECG data from CSV file."""
        try:
            print(f"Loading data from {self.csv_file}...")
            self.data = pd.read_csv(self.csv_file)
            
            # Check if we have the new time_seconds column or old timestamp column
            if 'time_seconds' in self.data.columns:
                # New format with time_seconds (already in seconds)
                self.time_col = 'time_seconds'
                if len(self.data) > 1:
                    duration = self.data[self.time_col].iloc[-1] - self.data[self.time_col].iloc[0]
                    self.sample_rate = len(self.data) / duration if duration > 0 else 500.0
                else:
                    duration = 0
                    self.sample_rate = 500.0
            else:
                # Legacy format with timestamp
                self.time_col = 'timestamp'
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                if len(self.data) > 1:
                    duration = (self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]).total_seconds()
                    self.sample_rate = len(self.data) / duration if duration > 0 else 500.0
                else:
                    duration = 0
                    self.sample_rate = 500.0
            
            print(f"Loaded {len(self.data)} samples")
            print(f"Estimated sample rate: {self.sample_rate:.1f} Hz")
            print(f"Duration: {duration:.1f} seconds")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def plot_channels(self, channels=None, time_window=None, start_time=None):
        """
        Plot ECG channels.
        
        Args:
            channels: List of channel numbers to plot (1-8), None for all
            time_window: Time window in seconds to plot, None for all data
            start_time: Start time offset in seconds, None for beginning
        """
        if self.data is None:
            self.load_data()
        
        # Default to all channels if not specified
        if channels is None:
            channels = list(range(1, 9))  # Channels 1-8
        
        # Filter data by time window if specified
        plot_data = self.data.copy()
        if time_window is not None or start_time is not None:
            start_idx = 0
            if start_time is not None:
                if self.time_col == 'time_seconds':
                    start_time_val = self.data[self.time_col].iloc[0] + start_time
                    start_idx = self.data[self.data[self.time_col] >= start_time_val].index[0]
                else:
                    start_timestamp = self.data['timestamp'].iloc[0] + pd.Timedelta(seconds=start_time)
                    start_idx = self.data[self.data['timestamp'] >= start_timestamp].index[0]
            
            if time_window is not None:
                if self.time_col == 'time_seconds':
                    end_time_val = plot_data[self.time_col].iloc[start_idx] + time_window
                    plot_data = plot_data[(plot_data[self.time_col] >= plot_data[self.time_col].iloc[start_idx]) & 
                                        (plot_data[self.time_col] <= end_time_val)]
                else:
                    end_timestamp = plot_data['timestamp'].iloc[start_idx] + pd.Timedelta(seconds=time_window)
                    plot_data = plot_data[(plot_data['timestamp'] >= plot_data['timestamp'].iloc[start_idx]) & 
                                        (plot_data['timestamp'] <= end_timestamp)]
            else:
                plot_data = plot_data[start_idx:]
        
        # Create time axis in seconds from start
        if self.time_col == 'time_seconds':
            time_seconds = plot_data[self.time_col] - plot_data[self.time_col].iloc[0]
        else:
            time_seconds = (plot_data['timestamp'] - plot_data['timestamp'].iloc[0]).dt.total_seconds()
        
        # Set up the plot
        n_channels = len(channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels), sharex=True)
        
        if n_channels == 1:
            axes = [axes]
        
        # Channel names for 12-lead ECG (using first 8)
        channel_names = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5']
        
        # Plot each channel
        for i, channel_num in enumerate(channels):
            if channel_num < 1 or channel_num > 8:
                print(f"Warning: Channel {channel_num} is out of range (1-8), skipping")
                continue
            
            channel_col = f'channel_{channel_num}'
            if channel_col not in plot_data.columns:
                print(f"Warning: {channel_col} not found in data, skipping")
                continue
            
            ax = axes[i]
            
            # Plot the ECG signal
            ax.plot(time_seconds, plot_data[channel_col], 'b-', linewidth=0.8, alpha=0.8)
            
            # Customize the subplot
            channel_name = channel_names[channel_num-1] if channel_num <= len(channel_names) else f'Ch{channel_num}'
            ax.set_ylabel(f'{channel_name}\n(mV)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(time_seconds.iloc[0], time_seconds.iloc[-1])
            
            # Add some basic statistics with high precision
            mean_val = plot_data[channel_col].mean()
            std_val = plot_data[channel_col].std()
            ax.set_title(f'Channel {channel_num} ({channel_name}) - Mean: {mean_val:.4f}mV, Std: {std_val:.4f}mV', 
                        fontsize=9, pad=5)
        
        # Set common x-label
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        
        # Add main title
        if self.time_col == 'time_seconds':
            start_time_str = f"T+{plot_data[self.time_col].iloc[0]:.3f}s"
        else:
            start_time_str = plot_data['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        duration = time_seconds.iloc[-1] - time_seconds.iloc[0]
        fig.suptitle(f'ECG Data - {self.csv_file.name}\nStart: {start_time_str}, Duration: {duration:.1f}s, '
                    f'Sample Rate: {self.sample_rate:.1f} Hz', fontsize=14, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
    
    def plot_overview(self):
        """Plot an overview of all channels in a compact format."""
        if self.data is None:
            self.load_data()
        
        # Create time axis in seconds
        if self.time_col == 'time_seconds':
            time_seconds = self.data[self.time_col] - self.data[self.time_col].iloc[0]
        else:
            time_seconds = (self.data['timestamp'] - self.data['timestamp'].iloc[0]).dt.total_seconds()
        
        # Create a 4x2 subplot layout for 8 channels
        fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
        axes = axes.flatten()
        
        channel_names = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5']
        
        for i in range(8):
            channel_col = f'channel_{i+1}'
            if channel_col in self.data.columns:
                ax = axes[i]
                ax.plot(time_seconds, self.data[channel_col], 'b-', linewidth=0.5, alpha=0.7)
                ax.set_title(f'{channel_names[i]} (Ch{i+1})', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_ylabel('mV', fontsize=8)
                
                if i >= 6:  # Bottom row
                    ax.set_xlabel('Time (s)', fontsize=8)
        
        # Main title
        if self.time_col == 'time_seconds':
            start_time_str = f"T+{self.data[self.time_col].iloc[0]:.3f}s"
        else:
            start_time_str = self.data['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        duration = time_seconds.iloc[-1]
        fig.suptitle(f'ECG Data Overview - {self.csv_file.name}\n'
                    f'Start: {start_time_str}, Duration: {duration:.1f}s, Samples: {len(self.data)}', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return fig
    
    def plot_statistics(self):
        """Plot basic statistics about the ECG data."""
        if self.data is None:
            self.load_data()
        
        # Calculate statistics for each channel
        channel_stats = {}
        for i in range(1, 9):
            channel_col = f'channel_{i}'
            if channel_col in self.data.columns:
                channel_stats[f'Ch{i}'] = {
                    'mean': self.data[channel_col].mean(),
                    'std': self.data[channel_col].std(),
                    'min': self.data[channel_col].min(),
                    'max': self.data[channel_col].max(),
                    'range': self.data[channel_col].max() - self.data[channel_col].min()
                }
        
        # Create statistics plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        channels = list(channel_stats.keys())
        
        # Mean values
        means = [channel_stats[ch]['mean'] for ch in channels]
        axes[0,0].bar(channels, means, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Mean Values by Channel')
        axes[0,0].set_ylabel('Mean (mV)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Standard deviation
        stds = [channel_stats[ch]['std'] for ch in channels]
        axes[0,1].bar(channels, stds, color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Standard Deviation by Channel')
        axes[0,1].set_ylabel('Std Dev (mV)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Min/Max ranges
        mins = [channel_stats[ch]['min'] for ch in channels]
        maxs = [channel_stats[ch]['max'] for ch in channels]
        x = np.arange(len(channels))
        axes[1,0].bar(x, maxs, color='lightgreen', alpha=0.7, label='Max')
        axes[1,0].bar(x, mins, color='orange', alpha=0.7, label='Min')
        axes[1,0].set_title('Min/Max Values by Channel')
        axes[1,0].set_ylabel('Value (mV)')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(channels, rotation=45)
        axes[1,0].legend()
        
        # Range (max - min)
        ranges = [channel_stats[ch]['range'] for ch in channels]
        axes[1,1].bar(channels, ranges, color='gold', alpha=0.7)
        axes[1,1].set_title('Range (Max - Min) by Channel')
        axes[1,1].set_ylabel('Range (mV)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.suptitle(f'ECG Data Statistics - {self.csv_file.name}', fontsize=14, y=0.98)
        plt.subplots_adjust(top=0.92)
        
        return fig


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Plot ECG data from CSV files')
    parser.add_argument('csv_file', help='CSV file containing ECG data')
    parser.add_argument('--channels', '-c', type=str, help='Channels to plot (e.g., "1,2,3" or "1-4")')
    parser.add_argument('--time-window', '-t', type=float, help='Time window in seconds to plot')
    parser.add_argument('--start-time', '-s', type=float, help='Start time offset in seconds')
    parser.add_argument('--overview', '-o', action='store_true', help='Show overview of all channels')
    parser.add_argument('--statistics', '--stats', action='store_true', help='Show statistics plots')
    parser.add_argument('--save', help='Save plot to file (specify filename)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File {args.csv_file} not found")
        sys.exit(1)
    
    # Create plotter
    plotter = ECGDataPlotter(args.csv_file)
    
    # Parse channels
    channels = None
    if args.channels:
        try:
            if '-' in args.channels:
                # Handle range like "1-4"
                start, end = map(int, args.channels.split('-'))
                channels = list(range(start, end + 1))
            else:
                # Handle list like "1,2,3"
                channels = [int(c.strip()) for c in args.channels.split(',')]
        except ValueError:
            print(f"Error: Invalid channel specification '{args.channels}'")
            sys.exit(1)
    
    # Create the appropriate plot
    if args.statistics:
        fig = plotter.plot_statistics()
    elif args.overview:
        fig = plotter.plot_overview()
    else:
        fig = plotter.plot_channels(channels=channels, 
                                  time_window=args.time_window, 
                                  start_time=args.start_time)
    
    # Save or show the plot
    if args.save:
        print(f"Saving plot to {args.save}")
        fig.savefig(args.save, dpi=300, bbox_inches='tight')
        print("Plot saved successfully")
    else:
        print("Displaying plot... Close the window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
