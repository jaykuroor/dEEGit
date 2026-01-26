from src.parse import parse, preprocess, events_to_tensor
from src.resampling_qc import compare_resampling_methods, plot_resampling_comparison, plot_raw_eeg


channels = ["AF3", "AF4", "T7", "T8", "PZ"]

# Parse raw data (before preprocessing)
raw_events = parse("data/IN.txt", min_len=200, database="IN", chan_num=5)

# Plot raw EEG signal before any preprocessing
plot_raw_eeg(raw_events, channels, event_idx=0, save_path='plots/')

# Preprocess and compare resampling methods
events = preprocess(raw_events, target_len=256, resample_method=1, chan_num=5)
results = compare_resampling_methods(events, channels, downsample_to=128, original_len=256)
plot_resampling_comparison(results, save_path='plots/')

# for method in [0,1,2]:  # 0: fft, 1: poly, 2: CubicSpline
#     print(f"\n=== Processing with {(['fft', 'poly', 'CubicSpline'][method])} resampling ===")
#     resevents = preprocess(events, target_len=256, resample_method=method, chan_num=5)
#     X, y = events_to_tensor(resevents, channels)
#     print(f"events stored: {len(resevents)}")
#     print(f"tensor X: {X.shape} {X.dtype}")
#     print(f"labels y: {y.shape} {y.dtype}")
    #plot_resampling_comparison(events, channels, resample_method=method)