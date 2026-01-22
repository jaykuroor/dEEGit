from src.parse import parse_events, events_to_tensor
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_resampling_comparison(events, channels, resample_method, save_dir="plots"):
    
    method_dir = os.path.join(save_dir, f"{resample_method}_resampling")
    os.makedirs(method_dir, exist_ok=True)
    
    method_label = "FFT" if resample_method == "fft" else "Polyphase"

    valid_events = [
        (eid, rec) for eid, rec in events.items()
        if all(ch in rec["channels"] for ch in channels)
    ]
    
    if not valid_events:
        print("No valid events found!")
        return

    valid_events.sort(key=lambda x: x[1]["original_len"])
    
    shortest_id, shortest_rec = valid_events[0]
    longest_id, longest_rec = valid_events[-1]
    
    for event_id, rec, label_suffix in [
        (shortest_id, shortest_rec, "shortest"),
        (longest_id, longest_rec, "longest")
    ]:
        orig_len = rec["original_len"]
        digit = rec["digit"]
        
        # --- Side-by-side comparison plot ---
        fig, axes = plt.subplots(5, 2, figsize=(14, 10))
        fig.suptitle(
            f"[{method_label}] Event {event_id} — Digit: {digit} — Original Length: {orig_len} ({label_suffix})",
            fontsize=14, fontweight='bold'
        )
        
        for i, ch_name in enumerate(channels):
            original_data = rec["original_channels"][ch_name]
            resampled_data = rec["channels"][ch_name]
            
            # og plot
            ax_orig = axes[i, 0]
            time_orig = np.arange(len(original_data))
            ax_orig.plot(time_orig, original_data, color='coral', linewidth=0.8)
            ax_orig.set_ylabel(ch_name, fontsize=10, rotation=0, labelpad=30, va='center')
            ax_orig.set_xlim(0, len(original_data) - 1)
            ax_orig.grid(True, alpha=0.3)
            ax_orig.tick_params(axis='y', labelsize=8)
            if i == 0:
                ax_orig.set_title(f"Original ({orig_len} samples)", fontsize=11)
            
            # resampled plpt
            ax_resamp = axes[i, 1]
            time_resamp = np.arange(len(resampled_data))
            ax_resamp.plot(time_resamp, resampled_data, color='steelblue', linewidth=0.8)
            ax_resamp.set_xlim(0, len(resampled_data) - 1)
            ax_resamp.grid(True, alpha=0.3)
            ax_resamp.tick_params(axis='y', labelsize=8)
            if i == 0:
                ax_resamp.set_title(f"Resampled - {method_label} (256 samples)", fontsize=11)
        
        axes[-1, 0].set_xlabel("Sample", fontsize=10)
        axes[-1, 1].set_xlabel("Sample", fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        filename = f"event_{event_id}_{label_suffix}_len{orig_len}_sidebyside.png"
        filepath = os.path.join(method_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")
        
        # --- Overlay comparison plot ---
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"[{method_label}] Event {event_id} — Digit: {digit} — Overlay Comparison ({label_suffix})",
            fontsize=14, fontweight='bold'
        )
        
        for i, ch_name in enumerate(channels):
            original_data = rec["original_channels"][ch_name]
            resampled_data = rec["channels"][ch_name]
            
            ax = axes[i]
            
            # og data
            time_orig = np.arange(len(original_data))
            ax.plot(time_orig, original_data, color='coral', linewidth=1.0, 
                   label=f'Original ({orig_len} samples)', alpha=0.7)
            
            # resampled with scaled x-axis to match og length
            time_resamp = np.linspace(0, len(original_data) - 1, len(resampled_data))
            ax.plot(time_resamp, resampled_data, color='steelblue', linewidth=1.0,
                   label=f'Resampled - {method_label} (256 samples)', alpha=0.7, linestyle='--')
            
            ax.set_ylabel(ch_name, fontsize=10, rotation=0, labelpad=30, va='center')
            ax.set_xlim(0, len(original_data) - 1)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelsize=8)
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)
        
        axes[-1].set_xlabel("Sample (original scale)", fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        filename = f"event_{event_id}_{label_suffix}_len{orig_len}_overlay.png"
        filepath = os.path.join(method_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")


channels = ["AF3", "AF4", "T7", "T8", "PZ"]

for method in ["fft", "poly"]:
    print(f"\n=== Processing with {method.upper()} resampling ===")
    events = parse_events("data/IN.txt", min_len=200, target_len=256, resample_method=method)
    X, y = events_to_tensor(events, channels)
    print(f"events stored: {len(events)}")
    print(f"tensor X: {X.shape} {X.dtype}")
    print(f"labels y: {y.shape} {y.dtype}")
    plot_resampling_comparison(events, channels, resample_method=method)