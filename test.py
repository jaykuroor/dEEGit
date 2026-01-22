from src.parse import parse_events, events_to_tensor

events = parse_events("data/IN.txt", min_len=250, target_len=256)

# Put your true 5 channels here in the order you want.
# If you don't care about order, sorted(...) is fine as long as it's consistent.
channels = ["AF3", "AF4", "T7", "T8", "PZ"]  # <- replace with your actual 5

X, y = events_to_tensor(events, channels)
print("events stored:", len(events))
print("tensor X:", X.shape, X.dtype)
print("labels y:", y.shape, y.dtype)
print("unique labels:", sorted(set(y.tolist()))[:15], "...")