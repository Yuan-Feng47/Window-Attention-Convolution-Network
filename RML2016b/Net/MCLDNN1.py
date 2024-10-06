import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from ptflops import get_model_complexity_info

class MCLDNN1(nn.Module):
    def __init__(self, num_classes=26):
        super(MCLDNN1, self).__init__()
        
        self.dr = 0.5  # dropout rate (%)

        self.conv1_1 = nn.Conv2d(2, 50, (1, 4), padding=(0,3))
        self.conv1_2 = nn.Conv1d(1, 50, 8, padding=3)
        self.conv1_3 = nn.Conv1d(1, 50, 8, padding=3)
        self.conv2 = nn.Conv2d(100, 50, (1, 4), padding=(0,3))
        self.conv4 = nn.Conv2d(100, 100, (1, 8), padding=0)
        self.relu1 = nn.ReLU()
        self.view = nn.Identity()  # 替换为 Identity 层

        self.lstm1 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dense = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1):
        x1 = input1

        input2 = input1[:, :1, :]
        input3 = input1[:, 1:, :]

        x1 = torch.unsqueeze(x1, 1)
        x1 = x1.reshape(-1, 2, 1, 128)
        
        x1 = self.conv1_1(x1)
        x1 = nn.ReLU()(x1)
        
        x2 = F.relu(self.conv1_2(input2))
        x3 = F.relu(self.conv1_3(input3))
        padding = nn.ConstantPad1d((0, 1), 0)
        x2 = padding(x2)
        x3 = padding(x3)

        x = torch.cat((x2, x3), dim=1)
        x = torch.unsqueeze(x, 1)

        x = torch.transpose(x, 0, 1)
        x = x.reshape(-1, 100, 1, 128)
        
        x = F.relu(self.conv2(x))
        x = padding(x)
        # print(x.shape)

       
        x1 = padding(x1)
        # x1 = padding1(x1)
        # print(x.shape)
        # print(x1.shape)

        x = torch.cat((x1, x), dim=1)
        # print(x.shape)
           
        x = F.relu(self.conv4(x))
        x = x.reshape(-1, 125, 100)
        # print(x.shape)

        x, _ = self.lstm1(x)

        x, _ = self.lstm2(x[:, -1, :])

        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=self.dr)
        x = F.selu(self.fc2(x))
        x = F.dropout(x, p=self.dr)
        
        x = self.dense(x)
        x = self.softmax(x)
        

        return x


# # Usage
# input_shape1 = [2, 1024]
# input_shape2 = [1024, 1]
# input_shape3 = [1024, 1]
# classes = 11
# model = MCLDNN1(classes)
# # Example usage:
# input1 = torch.rand(4, 2, 128)  # Example input with batch size 1
#
# output = model(input1)
# print(output)
# print(output.shape)
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
# MSCA = MCLDNN1().cuda()
#
# macsT, paramsT = compute_flops(MSCA, (dim, 1024))
#
#
# print('{:<30}  {:<8.6f}'.format('Computational complexity: ', macsT))
# print('{:<30}  {:<8.6f}'.format('Number of parameters: ', paramsT))
