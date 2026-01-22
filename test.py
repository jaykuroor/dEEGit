from src.parse import parse_events, events_to_tensor

events = parse_events("data/IN.txt", min_len=250, target_len=256)

channels = ["AF3", "AF4", "T7", "T8", "PZ"]

X, y = events_to_tensor(events, channels)
print("events stored:", len(events))
print("tensor X:", X.shape, X.dtype)
print("labels y:", y.shape, y.dtype)
print("unique labels:", sorted(set(y.tolist()))[:15], "...")