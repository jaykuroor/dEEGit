import numpy as np
from scipy.signal import resample, resample_poly
from scipy.interpolate import CubicSpline

def parse(path: str, min_len: int = 240, database: str = "IN", chan_num: int = 5) -> dict[int, dict]:
    
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
            elif int(parts[5]) < min_len or parts[2] != database:
                bad_events.add(event_id)
                events.pop(event_id, None)
                continue
            channel = parts[3]
            digit = int(parts[4])

            x = np.array(parts[6].split(","), dtype=np.float32)

            if x.size < min_len:
                bad_events.add(event_id)
                events.pop(event_id, None)
                continue

            rec = events.get(event_id)
            if rec is None:
                rec = {"digit": digit, "channels": {}}
                events[event_id] = rec
            else:

                if rec["digit"] != digit or channel in rec["channels"] or len(rec["channels"]) >= chan_num:
                    bad_events.add(event_id)
                    events.pop(event_id, None)
                    continue

            rec["channels"][channel] = x

    return events

def preprocess(events: dict[int, dict], target_len: int = 256, resample_method: int = 0, chan_num: int = 5) -> dict[int, dict]:
    new_events: dict[int, dict] = {}
    for event_id, rec in events.items():
        if len(rec["channels"]) < chan_num:
            continue
        ch_map = rec["channels"]
        new_ch_map: dict[str, np.ndarray] = {}
        for ch_name, x in ch_map.items():

            if x.size != target_len:

                if resample_method == 0:  # fft
                    x_resampled = resample(x, target_len).astype(np.float32)

                elif resample_method == 1:  # poly
                    x_resampled = resample_poly(x, target_len, x.size, padtype='line').astype(np.float32)

                elif resample_method == 2:  # CubicSpline
                    cs = CubicSpline(np.arange(x.size), x)
                    x_resampled = cs(np.linspace(0, x.size - 1, target_len)).astype(np.float32)

                elif resample_method == 3:  # None
                    x_resampled = x

                else:
                    raise ValueError(f"Invalid resample_method: {resample_method}")
                
                sd = np.std(x_resampled)

                if sd > 0:  x_normalized = (x_resampled - np.mean(x_resampled)) / sd
                else:       x_normalized = x_resampled - np.mean(x_resampled)

                new_ch_map[ch_name] = x_normalized
            else:
                sd = np.std(x)
                if sd > 0:  x_normalized = (x - np.mean(x)) / sd
                else:       x_normalized = x - np.mean(x)
                new_ch_map[ch_name] = x_normalized
        new_rec = {"digit": rec["digit"], "channels": new_ch_map}
        new_events[event_id] = new_rec
    return new_events

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
            
            
    
    