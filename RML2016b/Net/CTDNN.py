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
from ptflops import get_model_complexity_info
def compute_flops(model, input_shape):
    # 输入数据形状为 input_shape，元素值为随机数
    input_data = torch.randn((1, ) + input_shape).cuda()

    # 修改此处，使之能够接受 (dim, 1024) 形状的输入数据
    # 我们需要将 input_shape 作为一个元组传递给 get_model_complexity_info 函数
    macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                             print_per_layer_stat=True, verbose=True)
    return macs,params



dim = 2  # 根据你的实际输入数据的通道数设置
MSCA = CTDNN_net().cuda()

macsT, paramsT = compute_flops(MSCA, (dim, 1024))


print('{:<30}  {:<8.6f}'.format('Computational complexity: ', macsT))
print('{:<30}  {:<8.6f}'.format('Number of parameters: ', paramsT))
