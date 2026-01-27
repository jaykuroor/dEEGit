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
    
class _Unsqueeze(nn.Module):
    """Adds a dimension at `dim` (used to turn (B,C,T) into (B,1,C,T))."""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

class EEGNetBetter(nn.Module):

    def __init__(
        self,
        chan_num=5,
        time_len=256,
        num_classes=10,
        F1=8,              # number of temporal filters
        D=2,               # number of spatial filters per temporal filter
        F2=None,           # number of pointwise filters after separable conv (default F1*D)
        kernel_length=64,  # temporal kernel length
        sep_length=16,     # separable (depthwise temporal) kernel length
        dropout=0.25
    ):
        super().__init__()

        if F2 is None:
            F2 = F1 * D  # common EEGNet choice :contentReference[oaicite:1]{index=1}

        # 1) reshape (B,C,T) -> (B,1,C,T)
        self.prep = nn.Sequential(
            _Unsqueeze(dim=1)
        )

        # 2) Temporal conv: learns F1 time filters (like data-driven band-pass filters)
        # Kernel is (1, kernel_length): only slides over time, not channels.
        self.temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False
            ),
            # EEGNet references often use eps=1e-3, momentum=0.01 :contentReference[oaicite:2]{index=2}
            nn.BatchNorm2d(F1, eps=1e-3, momentum=0.01),
        )

        # 3) Spatial filtering (the key EEGNet step):
        # Depthwise conv spanning ALL channels: kernel (C,1), groups=F1
        # Produces D spatial filters per temporal filter (so output channels = F1*D)
        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(chan_num, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F1 * D, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # 4) Separable conv refinement:
        # depthwise temporal conv (per feature map) then pointwise (1x1) mixing
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
            nn.BatchNorm2d(F2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # 5) Classifier: infer the flatten size safely (no hard-coded math)
        with torch.no_grad():
            dummy = torch.zeros(1, chan_num, time_len)
            feat = self.refine(self.spatial(self.temporal(self.prep(dummy))))
            flat_dim = feat.flatten(1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.refine(x)
        x = self.fc_layers(x)
        return x