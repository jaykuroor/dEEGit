import numpy as np
from scipy.signal import resample, resample_poly

def parse_events(path: str, target_len: int = 256, min_len: int = 240, resample_method: str = "fft") -> dict[int, dict]:
    
    if resample_method not in ("fft", "poly"):
        raise ValueError(f"resample_method must be 'fft' or 'poly', got '{resample_method}'")
    
    events: dict[int, dict] = {}
    bad_events: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.rstrip().split("\t")
            if len(parts) < 7:
                continue

            event_id = int(parts[1])
            if event_id in bad_events:
                continue
            channel = parts[3]
            digit = int(parts[4])

            x = np.fromstring(parts[6], sep=",", dtype=np.float32)

            if x.size < min_len:
                bad_events.add(event_id)
                events.pop(event_id, None)
                continue

            original_x = x.copy()
            original_len = x.size
            
            if x.size != target_len:
                if resample_method == "fft":
                    x = resample(x, target_len).astype(np.float32)
                else:  # poly
                    x = resample_poly(x, target_len, x.size, padtype='line').astype(np.float32)

            rec = events.get(event_id)
            if rec is None:
                rec = {"digit": digit, "channels": {}, "original_channels": {}, "original_len": original_len}
                events[event_id] = rec
            else:

                if rec["digit"] != digit:
                    bad_events.add(event_id)
                    events.pop(event_id, None)
                    continue

            rec["channels"][channel] = x
            rec["original_channels"][channel] = original_x

    return events

def events_to_tensor(
    events: dict[int, dict],
    channels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build:
      X: (N, 5, 256) float32
      y: (N,) int64
    Keeps only events that have all required channels.
    """
    X_list = []
    y_list = []

    for rec in events.values():
        ch_map = rec["channels"]
        if all(ch in ch_map for ch in channels):
            X_list.append(np.stack([ch_map[ch] for ch in channels], axis=0))  # (5,256)
            y_list.append(rec["digit"])

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N,5,256)
    y = np.array(y_list, dtype=np.int64)
    return X, y
            
            
    
    