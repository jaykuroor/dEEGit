import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from parse import parse, preprocess, events_to_tensor
import config

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def load_data(resample_method=1):
    events = parse(
        config.DATA_PATH,
        min_len=config.MIN_LEN,
        database=config.DATABASE,
        chan_num=config.CHAN_NUM
    )
    events = preprocess(events, target_len=config.TARGET_LEN, resample_method=resample_method, chan_num=config.CHAN_NUM)
    X, y = events_to_tensor(events, config.CHANNELS)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - config.TRAIN_SPLIT), random_state=config.SEED, stratify=y
    )
    val_size = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), random_state=config.SEED, stratify=y_temp
    )

    return(
        EEGDataset(X_train, y_train),
        EEGDataset(X_val, y_val),
        EEGDataset(X_test, y_test)
    )