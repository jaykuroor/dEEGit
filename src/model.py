import torch
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

class EEGNetBetter(nn.Module):

    def __init__(
        self,
        chan_num=5,
        time_len=256,
        num_classes=10,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        sep_length=16,
        dropout=0.25
    ):
        super().__init__()
        
        self.temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False
            ),
            nn.BatchNorm2d(F1),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(chan_num, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, sep_length),
                padding=(0, sep_length // 2),
                groups=F1 * D,
                bias=False
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, chan_num, time_len)
            x = dummy.unsqueeze(1)
            x = self.temporal(x)
            x = self.spatial(x)
            feat = self.refine(x)
            flat_dim = feat.flatten(1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # convert to (B,1,C,T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.refine(x)
        x = self.fc_layers(x)
        return x
