import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ptflops import get_model_complexity_info
from torchsummary import summary

# 定义自定义的CausalPadding类
class CausalPadding(nn.Module):
    def __init__(self, kernel_size):
        super(CausalPadding, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # 在序列的左侧进行padding
        padding = torch.zeros(x.size(0), x.size(1), self.kernel_size - 1).to(x.device)
        x = torch.cat((padding, x), dim=2)
        return x

class MCLDNN(nn.Module):
    def __init__(self, num_classes=26):
        super(MCLDNN, self).__init__()

        self.dr = 0.5  # dropout rate (%)
        #I/O两路
        self.conv1_1 = nn.Conv2d(2, 50, (2, 8), padding='same')
        #分别I和O
        self.conv1_2 = nn.Sequential(
            CausalPadding(8),
            nn.Conv1d(1, 50, 8)
        )
        self.conv1_3 = nn.Sequential(
            CausalPadding(8),
            nn.Conv1d(1, 50, 8)
        )
        #I和O结果加在一起再来一次
        self.conv2 = nn.Conv2d(100, 50, (1, 8), padding='same')
        #结果和I/O两路的conv1加在一起再来一次
        self.conv4 = nn.Conv2d(100, 100, (1, 5), padding='valid')
        self.relu1 = nn.ReLU()
        self.view = nn.Identity()  # 替换为 Identity 层

        self.lstm1 = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.softmax = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(128, num_classes)

        # 初始化权重
        nn.init.xavier_uniform_(self.conv1_1.weight)
        nn.init.xavier_uniform_(self.conv1_2[1].weight)
        nn.init.xavier_uniform_(self.conv1_3[1].weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

    def forward(self, input1):
        # 预处理
        x1=input1
        # 使用切片操作将 input1 分割成 input2 和 input3
        input2 = input1[ :,:1, :]
        input3 = input1[:, 1:, :]

        x1= torch.unsqueeze(x1, 1)  # 将输入数据的维度调整为 [batch_size, 1, 1024]
        x1 = x1.reshape(-1,2,1, 1024)

        x1 = self.conv1_1(x1)
        x1 = nn.ReLU()(x1)

        x2 = F.relu(self.conv1_2(input2))
        x3 = F.relu(self.conv1_3(input3))
        x = torch.cat((x2, x3), dim=1)
        x = torch.unsqueeze(x, 1)  # 在第1维插入一个维度

        x = torch.transpose(x, 0, 1)  # 将第1维和第2维进行交换
        x = x.reshape(-1,100, 1,1024)

        x = F.relu(self.conv2(x))

        x = torch.cat((x1, x), dim=1)

        x = F.relu(self.conv4(x))

        x = x.reshape(-1, 1020, 100)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x[:, -1, :])

        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=self.dr)
        x = F.selu(self.fc2(x))
        x = F.dropout(x, p=self.dr)
        x = self.fc3(x)
        x= self.softmax(x)

        return x



        


# model = MCLDNN()

# # Print the model architecture
# print(model)

# # Save the model
# torch.save(model.state_dict(), 'model.pth')



# # Usage
# input_shape1 = [2, 1024]
# input_shape2 = [1024, 1]
# inout_shape3 = [1024, 1]
# classes = 11
# model = MCLDNN( classes)
# # Example usage:
# input1 = torch.rand(500,2, 128)  # Example input with batch size 1


# output = model(input1)
# print(output)
# def compute_flops(model, input_shape):
#     # 输入数据形状为 input_shape，元素值为随机数
#     input_data = torch.randn((1, ) + input_shape).cuda()

#     # 修改此处，使之能够接受 (dim, 1024) 形状的输入数据
#     # 我们需要将 input_shape 作为一个元组传递给 get_model_complexity_info 函数
#     macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
#                                              print_per_layer_stat=True, verbose=True)
#     return macs,params



# dim = 2  # 根据你的实际输入数据的通道数设置
# MSCA = MCLDNN().cuda()

# macsT, paramsT = compute_flops(MSCA, (dim, 1024))


# print('{:<30}  {:<8.6f}'.format('Computational complexity: ', macsT))
# print('{:<30}  {:<8.6f}'.format('Number of parameters: ', paramsT))




# model = MCLDNN(num_classes=26)
# # 打印模型信息
# network_name = type(model).__name__
# print("Network name:", network_name)
# # 打印模型结构
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# summary(model, (2, 1024))
