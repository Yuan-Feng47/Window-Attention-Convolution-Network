import torch
import torch.nn as nn
import torch.nn.functional as F


class CTDNN(nn.Module):
    def __init__(self, num_classes):
        super(CTDNN, self).__init__()
        # CNN Backbone
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=15, padding=7),  # Adjusted for 2 input channels
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, padding=7),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        # Transition Module
        self.transition = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)

        # Transformer Module
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=3
        )

        # Classifier
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.transition(x)
        x = x.permute(2, 0, 1)  # Adjust shape for transformer
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.classifier(x)
        return x



def CTDNN_net(**kwargs):
    model = CTDNN(num_classes=24)
    return model
