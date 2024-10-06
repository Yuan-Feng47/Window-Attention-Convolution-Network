import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info

class CNNModel(nn.Module):
    def __init__(self, num_classes=26, dropout_rate=0.2):
        super(CNNModel, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 64 * 8)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=26):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64 * 256, 512)  # The dimension here needs to be adjusted according to your dataset.
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


# model = CNN(num_classes=26)
# # 打印模型信息
# network_name = type(model).__name__
# print("Network name:", network_name)
# # 打印模型结构
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# summary(model, (2, 1024))

# def compute_flops(model, input_shape):
#     # 输入数据形状为 input_shape，元素值为随机数
#     input_data = torch.randn((1, ) + input_shape).cuda()
#
#     # 修改此处，使之能够接受 (dim, 1024) 形状的输入数据
#     # 我们需要将 input_shape 作为一个元组传递给 get_model_complexity_info 函数
#     macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
#                                              print_per_layer_stat=True, verbose=True)
#     return macs,params
#
#
#
# dim = 2  # 根据你的实际输入数据的通道数设置
# MSCA = CNN().cuda()
#
# macsT, paramsT = compute_flops(MSCA, (dim, 1024))
#
#
# print('{:<30}  {:<8.6f}'.format('Computational complexity: ', macsT))
# print('{:<30}  {:<8.6f}'.format('Number of parameters: ', paramsT))