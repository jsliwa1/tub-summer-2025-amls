import torch
from torch import nn

class ResidualBlock1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size//2, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity()
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    

class ECGClassifier(nn.Module):
    
    def __init__(self, num_classes: int=4, res_blocks: int=5, channels: int=32,
                 lstm_hidden_size: int=64, fc_hidden_dim: int=64, dropout_proba: float=0.3):
        super(ECGClassifier, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(channels)
        )
        blocks = []
        for _ in range(res_blocks):
            blocks.append(ResidualBlock1d(channels, channels))
            blocks.append(nn.MaxPool1d(kernel_size=5, stride=2))
        self.res_blocks = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(input_size=channels, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_proba),
            nn.Linear(fc_hidden_dim, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = torch.mean(lstm_out, dim=1)
        return self.fc(x)