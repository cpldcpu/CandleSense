#!/usr/bin/env python3
"""
Analyze collected capacitive flicker sensor data
Usage: python3 analyze_data.py <csv_file>
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, stddev, offset):
    """Gaussian function for peak fitting"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

def fit_gaussian_peak(freqs, psd, peak_idx, window_size=10):
    """
    Fit a Gaussian to a peak in the PSD
    
    Args:
        freqs: frequency array
        psd: power spectral density array
        peak_idx: index of the peak maximum
        window_size: number of points around peak to include in fit
        
    Returns:
        fitted_freq: refined peak frequency
        fitted_params: [amplitude, mean, stddev, offset]
        fit_success: boolean indicating if fit was successful
    """
    # Define fitting window around the peak
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(freqs), peak_idx + window_size + 1)
    
    freq_window = freqs[start_idx:end_idx]
    psd_window = psd[start_idx:end_idx]
    
    if len(freq_window) < 5:  # Need minimum points for fitting
        return freqs[peak_idx], None, False
    
    # Initial parameter guesses
    amplitude_guess = psd[peak_idx] - np.min(psd_window)
    mean_guess = freqs[peak_idx]
    stddev_guess = (freq_window[-1] - freq_window[0]) / 6  # rough estimate
    offset_guess = np.min(psd_window)
    
    initial_guess = [amplitude_guess, mean_guess, stddev_guess, offset_guess]
    
    try:
        # Perform the fit
        popt, pcov = curve_fit(gaussian, freq_window, psd_window, p0=initial_guess, maxfev=1000)
        fitted_freq = popt[1]  # mean parameter is the peak frequency
        
        # Check if fit is reasonable (peak within original window)
        if freq_window[0] <= fitted_freq <= freq_window[-1]:
            return fitted_freq, popt, True
        else:
            return freqs[peak_idx], None, False
            
    except (RuntimeError, ValueError):
        # Fitting failed, return original peak
        return freqs[peak_idx], None, False

if len(sys.argv) < 2:
    print("Usage: python3 analyze_data.py <csv_file>")
    sys.exit(1)

# Load data
df = pd.read_csv(sys.argv[1])
print(f"Loaded {len(df)} samples")
print(f"Columns: {list(df.columns)}")

# Convert timestamp to datetime if present
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Capacitive Flicker Sensor Analysis', fontsize=16)

# Plot 1: Raw data and average over time
axes[0].plot(df.index, df['raw'], 'b-', alpha=0.7, linewidth=0.5, label='Raw')
axes[0].plot(df.index, df['avg'] // 32, 'r-', alpha=0.8, linewidth=0.8, label='Average (÷32)')
axes[0].set_title('Raw ADC Values & Filtered Average')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('ADC Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: flicker detection signal with zero crossings
axes[1].plot(df.index, df['hp'], 'g-', alpha=0.7, linewidth=0.5, label='Flicker Signal')
zero_cross_indices = df.index[df['zero_cross'] == 1]
if len(zero_cross_indices) > 0:
    axes[1].scatter(zero_cross_indices, df.loc[zero_cross_indices, 'hp'], 
                   color='red', s=30, marker='x', label='Zero Crossings', zorder=5, linewidth=2)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1].set_title('Flicker Detection Signal + Zero Crossings')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('Flicker Delta')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Frequency analysis of raw signal
# Deduce sample frequency from timestamps
if 'time_seconds' in df.columns and len(df) > 1:
    total_time = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]
    sample_rate = (len(df) - 1) / total_time if total_time > 0 else 8000
    print(f"Deduced sample rate: {sample_rate:.1f} Hz")
else:
    sample_rate = 80  # Fallback if no timestamp data
    print(f"Using fallback sample rate: {sample_rate} Hz")

freqs, psd = signal.welch(df['raw'], fs=sample_rate, nperseg=1024)
axes[2].semilogy(freqs, psd)

# Find the maximum peak above 3Hz
freq_mask = freqs > 3.0  # Only consider frequencies above 3Hz
if np.any(freq_mask):
    freqs_above_3hz = freqs[freq_mask]
    psd_above_3hz = psd[freq_mask]
    
    # Find the maximum peak above 3Hz
    max_peak_idx_local = np.argmax(psd_above_3hz)
    max_peak_idx_global = np.where(freq_mask)[0][max_peak_idx_local]  # Convert to global index
    max_peak_freq_raw = freqs_above_3hz[max_peak_idx_local]
    max_peak_psd = psd_above_3hz[max_peak_idx_local]
    
    # Perform Gaussian peak fitting
    fitted_freq, fitted_params, fit_success = fit_gaussian_peak(freqs, psd, max_peak_idx_global, window_size=15)
    
    if fit_success:
        print(f"Raw peak: {max_peak_freq_raw:.2f} Hz")
        print(f"Gaussian fitted peak: {fitted_freq:.3f} Hz (PSD: {max_peak_psd:.2e})")
        print(f"Peak fitting improvement: {abs(fitted_freq - max_peak_freq_raw):.3f} Hz")
        
        # Plot both raw and fitted peaks
        axes[2].plot(max_peak_freq_raw, max_peak_psd, 'bo', markersize=8, label=f'Raw peak: {max_peak_freq_raw:.1f} Hz')
        axes[2].plot(fitted_freq, gaussian(fitted_freq, *fitted_params), 'ro', markersize=8, 
                    label=f'Fitted peak: {fitted_freq:.3f} Hz')
        axes[2].axvline(x=fitted_freq, color='red', linestyle='--', alpha=0.7)
        
        # Optionally plot the fitted Gaussian curve
        if fitted_params is not None:
            # Create fine frequency grid around the peak for smooth curve
            freq_fine = np.linspace(fitted_freq - 2, fitted_freq + 2, 100)
            freq_fine = freq_fine[(freq_fine >= freqs[0]) & (freq_fine <= freqs[-1])]
            if len(freq_fine) > 0:
                gaussian_curve = gaussian(freq_fine, *fitted_params)
                axes[2].plot(freq_fine, gaussian_curve, 'r--', alpha=0.6, linewidth=1, label='Gaussian fit')
        
        peak_freq_final = fitted_freq
    else:
        print(f"Gaussian fitting failed, using raw peak: {max_peak_freq_raw:.2f} Hz (PSD: {max_peak_psd:.2e})")
        axes[2].plot(max_peak_freq_raw, max_peak_psd, 'ro', markersize=8, label=f'Max peak: {max_peak_freq_raw:.1f} Hz')
        axes[2].axvline(x=max_peak_freq_raw, color='red', linestyle='--', alpha=0.7)
        peak_freq_final = max_peak_freq_raw
    
    axes[2].legend()
else:
    print("No frequency data above 3Hz available")
    peak_freq_final = None

axes[2].set_title(f'Power Spectral Density (Raw) - Fs={sample_rate:.0f}Hz')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('PSD')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Print statistics
print(f"\nStatistics:")
print(f"Raw signal - Mean: {df['raw'].mean():.1f}, Std: {df['raw'].std():.1f}")
print(f"Flicker signal - Mean: {df['hp'].mean():.1f}, Std: {df['hp'].std():.1f}")
print(f"Zero crossings - Total: {df['zero_cross'].sum()}, Rate: {df['zero_cross'].sum()/len(df)*100:.2f}%")

# Detect Flicker events (simple threshold)
flicker_threshold = df['hp'].std() * 3
flicker_events = df['hp'] > flicker_threshold
print(f"flicker events detected: {flicker_events.sum()} (threshold: {flicker_threshold:.1f})")

if peak_freq_final is not None:
    print(f"Final peak frequency: {peak_freq_final:.3f} Hz")

plt.show()
