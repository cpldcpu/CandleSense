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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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

def plot_psd_with_gaussian(data, fs, ax, label_base, curve_color, peak_color,
                           min_freq=3.0, window_size=15, nperseg=1024,
                           title_suffix=''):
    """
    Compute PSD, fit Gaussian to dominant peak above min_freq, and plot.
    Returns: peak frequency (float) or None.
    """
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    ax.semilogy(freqs, psd, color=curve_color, label=f'{label_base} PSD')
    freq_mask = freqs > min_freq
    peak_freq_final = None
    if np.any(freq_mask):
        freqs_sel = freqs[freq_mask]
        psd_sel = psd[freq_mask]
        local_idx = np.argmax(psd_sel)
        global_idx = np.where(freq_mask)[0][local_idx]
        raw_peak_freq = freqs_sel[local_idx]
        raw_peak_val = psd_sel[local_idx]
        fitted_freq, fitted_params, fit_success = fit_gaussian_peak(freqs, psd, global_idx, window_size=window_size)
        if fit_success:
            print(f"{label_base} raw peak: {raw_peak_freq:.2f} Hz")
            print(f"{label_base} Gaussian fitted peak: {fitted_freq:.3f} Hz (PSD: {raw_peak_val:.2e})")
            print(f"{label_base} peak fitting improvement: {abs(fitted_freq - raw_peak_freq):.3f} Hz")
            ax.plot(fitted_freq, gaussian(fitted_freq, *fitted_params), marker='o',
                    color=peak_color, markersize=7, label=f'{label_base} fitted peak: {fitted_freq:.3f} Hz')
            ax.axvline(fitted_freq, color=peak_color, linestyle='--', alpha=0.7)
            # Gaussian curve
            freq_fine = np.linspace(fitted_freq - 2, fitted_freq + 2, 200)
            freq_fine = freq_fine[(freq_fine >= freqs[0]) & (freq_fine <= freqs[-1])]
            if len(freq_fine):
                ax.plot(freq_fine, gaussian(freq_fine, *fitted_params),
                        linestyle='--', color=peak_color, alpha=0.6, linewidth=1,
                        label=f'{label_base} Gaussian fit')
            peak_freq_final = fitted_freq
        else:
            print(f"{label_base} Gaussian fitting failed, using raw peak: {raw_peak_freq:.2f} Hz (PSD: {raw_peak_val:.2e})")
            ax.plot(raw_peak_freq, raw_peak_val, marker='o', color=peak_color,
                    markersize=7, label=f'{label_base} peak: {raw_peak_freq:.2f} Hz')
            ax.axvline(raw_peak_freq, color=peak_color, linestyle='--', alpha=0.7)
            peak_freq_final = raw_peak_freq
        ax.legend()
    else:
        print(f"No {label_base} frequency data above {min_freq} Hz available")
    ax.set_title(f'PSD ({label_base}){title_suffix}')
    ax.grid(True, alpha=0.3)
    return peak_freq_final

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
fig = plt.figure(figsize=(18, 6))
fig.suptitle('Capacitive Flicker Sensor Analysis', fontsize=16)
gs = GridSpec(1, 3, width_ratios=[1, 1, 1], figure=fig)
ax_raw = fig.add_subplot(gs[0, 0])
ax_flicker = fig.add_subplot(gs[0, 1])

# Nested grid for PSD (third column split vertically)
gs_psd = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.05)
ax_psd_raw = fig.add_subplot(gs_psd[0, 0])
ax_psd_hp = fig.add_subplot(gs_psd[1, 0], sharex=ax_psd_raw)

# Plot 1: Raw data and average over time
ax_raw.plot(df.index, df['raw'], 'b-', alpha=0.7, linewidth=0.5, label='Raw')
ax_raw.plot(df.index, df['avg'] // 32, 'r-', alpha=0.8, linewidth=0.8, label='Average (÷32)')
ax_raw.set_title('Raw ADC Values & Filtered Average')
ax_raw.set_xlabel('Sample')
ax_raw.set_ylabel('ADC Value')
ax_raw.legend()
ax_raw.grid(True, alpha=0.3)

# Plot 2: flicker detection signal with zero crossings
ax_flicker.plot(df.index, df['hp'], 'g-', alpha=0.7, linewidth=0.5, label='Flicker Signal')
zero_cross_indices = df.index[df['zero_cross'] == 1]
if len(zero_cross_indices) > 0:
    ax_flicker.scatter(
        zero_cross_indices,
        df.loc[zero_cross_indices, 'hp'],
        color='red',
        s=30,
        marker='x',
        label='Zero Crossings',
        zorder=5,
        linewidth=2
    )
ax_flicker.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax_flicker.set_title('Flicker Detection Signal + Zero Crossings')
ax_flicker.set_xlabel('Sample')
ax_flicker.set_ylabel('Flicker Delta')
ax_flicker.legend()
ax_flicker.grid(True, alpha=0.3)

# Frequency analysis (sample rate deduction)
if 'time_seconds' in df.columns and len(df) > 1:
    total_time = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]
    sample_rate = (len(df) - 1) / total_time if total_time > 0 else 8000
    print(f"Deduced sample rate: {sample_rate:.1f} Hz")
else:
    sample_rate = 80
    print(f"Using fallback sample rate: {sample_rate} Hz")

# REPLACED duplicated PSD + peak fitting blocks with helper calls
peak_freq_final = plot_psd_with_gaussian(
    df['raw'], sample_rate, ax_psd_raw,
    label_base='Raw',
    curve_color='C0',
    peak_color='red',
    title_suffix=f' - Fs={sample_rate:.0f}Hz'
)
ax_psd_raw.set_ylabel('PSD (Raw)')
plt.setp(ax_psd_raw.get_xticklabels(), visible=False)

hp_peak_freq_final = plot_psd_with_gaussian(
    df['hp'], sample_rate, ax_psd_hp,
    label_base='HP',
    curve_color='g',
    peak_color='magenta'
)
ax_psd_hp.set_xlabel('Frequency (Hz)')
ax_psd_hp.set_ylabel('PSD (HP)')

plt.tight_layout(rect=[0, 0, 1, 0.95])

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
    print(f"Final raw peak frequency: {peak_freq_final:.3f} Hz")
if hp_peak_freq_final is not None:
    print(f"Final HP peak frequency: {hp_peak_freq_final:.3f} Hz")

plt.show()
