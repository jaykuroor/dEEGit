from scipy.signal import resample, resample_poly
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

def compare_resampling_methods(
    events: dict[int, dict],
    channels: list[str],
    downsample_to: int = 128,
    original_len: int = 256,
    scount: int = 100
) -> dict[str, dict]:

    results = {
        'fft': {'mse': [], 'residues': []},
        'poly': {'mse': [], 'residues': []},
        'cubic': {'mse': [], 'residues': []}
    }
    
    for rec in events.values():
        if scount < 0:
            break
        ch_map = rec["channels"]
        if not all(ch in ch_map for ch in channels):
            continue
            
        for ch_name in channels:
            original = ch_map[ch_name]
            if original.size != original_len:
                continue

            step = original_len // downsample_to
            downsampled = original[::step][:downsample_to]  # decimation
            

            # FFT
            reconstructed_fft = resample(downsampled, original_len).astype(np.float32)
            mse_fft = np.mean((original - reconstructed_fft) ** 2)
            results['fft']['mse'].append(mse_fft)
            results['fft']['residues'].append(original - reconstructed_fft)
            
            # Poly
            reconstructed_poly = resample_poly(downsampled, original_len, downsample_to, padtype='line').astype(np.float32)
            mse_poly = np.mean((original - reconstructed_poly) ** 2)
            results['poly']['mse'].append(mse_poly)
            results['poly']['residues'].append(original - reconstructed_poly)
            
            # Cubic
            cs = CubicSpline(np.arange(downsample_to), downsampled)
            reconstructed_cubic = cs(np.linspace(0, downsample_to - 1, original_len)).astype(np.float32)
            mse_cubic = np.mean((original - reconstructed_cubic) ** 2)
            results['cubic']['mse'].append(mse_cubic)
            results['cubic']['residues'].append(original - reconstructed_cubic)
            scount -= 1
    return results

def plot_resampling_comparison(results: dict, save_path: str = 'plots/'):
    """
    Create comprehensive plots comparing resampling methods.
    """
    methods = ['fft', 'poly', 'cubic']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # box plot of MSE
    mse_data = [results[method]['mse'] for method in methods]
    axes[0].boxplot(mse_data, labels=['FFT', 'Poly', 'Cubic'])
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE Distribution Across Methods')
    axes[0].grid(True, alpha=0.3)
    
    for method in methods:
        all_residues = np.concatenate(results[method]['residues'])
        axes[1].hist(all_residues, bins=50, alpha=0.5, label=method.upper())
    axes[1].set_xlabel('Residue Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residue Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}resampling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # summary stats
    for method in methods:
        print(f"\n{method.upper()}:")
        print(f"  Mean MSE: {np.mean(results[method]['mse']):.6f}")
        all_res = np.concatenate(results[method]['residues'])
        print(f"  Mean Residue: {np.mean(np.abs(all_res)):.6f}")
        print(f"  Std Residue:  {np.std(all_res):.6f}")


def plot_raw_eeg(
    events: dict[int, dict],
    channels: list[str],
    event_idx: int = 0,
    save_path: str = 'plots/'
):
    """
    Plot raw EEG data showing all 5 channels stacked vertically for one event.
    Shows what the data looks like before any preprocessing.
    """
    # Get the specified event
    event_ids = list(events.keys())
    if event_idx >= len(event_ids):
        print(f"Event index {event_idx} out of range. Using first event.")
        event_idx = 0
    
    event_id = event_ids[event_idx]
    rec = events[event_id]
    digit = rec["digit"]
    ch_map = rec["channels"]
    
    # Create figure with subplots for each channel
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Raw EEG Signal - Event #{event_id} (Digit: {digit})', fontsize=14, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Distinct colors
    
    for idx, ch_name in enumerate(channels):
        if ch_name in ch_map:
            signal = ch_map[ch_name]
            time_ms = np.arange(len(signal)) / 128 * 1000  # Convert to milliseconds (128 Hz)
            
            axes[idx].plot(time_ms, signal, color=colors[idx], linewidth=0.8)
            axes[idx].set_ylabel(f'{ch_name}\n(μV)', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(0, time_ms[-1])
            
            # Add sample count annotation
            axes[idx].text(0.98, 0.95, f'n={len(signal)} samples', 
                          transform=axes[idx].transAxes, fontsize=9,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[idx].text(0.5, 0.5, f'{ch_name}: No data', 
                          transform=axes[idx].transAxes, ha='center', va='center')
    
    axes[-1].set_xlabel('Time (ms)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}raw_eeg_signal.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved raw EEG plot to {save_path}raw_eeg_signal.png")