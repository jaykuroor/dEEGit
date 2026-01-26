import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, chan_num=5, time_len=256, num_classes=10):
        super(EEGNet, self).__init__()

        filter_size = chan_num * 8

        self.temporal = nn.Sequential(
            nn.Conv1d(chan_num, filter_size, kernel_size=64, groups=chan_num, padding=32, bias=False),
            nn.BatchNorm1d(filter_size),
            )

        self.spatial = nn.Sequential(
            nn.Conv1d(filter_size, filter_size, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(filter_size),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.2),
        )

        self.refine = nn.Sequential(
            nn.Conv1d(filter_size, filter_size, kernel_size=16, groups=filter_size, padding=8, bias=False),
            nn.Conv1d(filter_size, filter_size, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(filter_size),
            nn.ELU(),
            nn.AvgPool1d(8),
            nn.Dropout(0.3),

        )

        conv_output_len = 40 * (time_len // (4*8))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_len, num_classes)
        )

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.refine(x)
        x = self.fc_layers(x)
        return x