import torch
import torch.nn as nn
from torchsummary import summary

class GaussianNoise(nn.Module):
    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x



class ICAMC(nn.Module):
    def __init__(self, input_shape=[2, 1024],num_classes=11):
        super(ICAMC, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size= 8, padding='same')
        self.maxpool1 = nn.MaxPool1d(kernel_size=2 ,stride=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size= 4, padding='same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8,  padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8,  padding='same')
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        flattened_shape = self._get_flattened_shape(input_shape)
        
     

        self.dense1 = nn.Linear(flattened_shape , 128)
        self.gaussian_noise = GaussianNoise(1)
        self.dense2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x = x.unsqueeze(1)
    
        x = self.conv1(x)
        
        x = nn.ReLU()(x)
        
        x = self.maxpool1(x)
        x = self.conv2(x)
        
        x = nn.ReLU()(x)
    
        x = self.conv3(x)
        
        x = nn.ReLU()(x)
        
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv4(x)
        
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.flatten(x)
    
    
        x = self.dense1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        noise = torch.randn_like(x) * 1  # 创建新的高斯噪声张量
        x = x + noise  # 将高斯噪声添加到输入中
        x = self.dense2(x)
        x = self.softmax(x)
        return x

    def _get_flattened_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        #x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        flattened_shape = self.flatten(x).shape[1]
        return flattened_shape



# # Create an instance of the model
# model = ICAMC()
#
# # Print the model architecture
# print(model)
#
# # 打印模型信息
# network_name = type(model).__name__
# print("Network name:", network_name)
# # 打印模型结构
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# summary(model, ( 2, 128))