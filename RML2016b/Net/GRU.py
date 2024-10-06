import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

class GRUModel(nn.Module):
    def __init__(self, num_classes=26):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take the last hidden state of the GRU
        x = self.fc(x)
        x = self.softmax(x)
        return x
    

# def compute_flops(model, input_shape):
#     # 输入数据形状为 input_shape，元素值为随机数
#     input_data = torch.randn((1, ) + input_shape).cuda()

#     # 修改此处，使之能够接受 (dim, 1024) 形状的输入数据
#     # 我们需要将 input_shape 作为一个元组传递给 get_model_complexity_info 函数
#     macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
#                                              print_per_layer_stat=True, verbose=True)
#     return macs,params



# dim = 2  # 根据你的实际输入数据的通道数设置
# MSCA = GRUModel().cuda()

# macsT, paramsT = compute_flops(MSCA, (dim, 128))


# print('{:<30}  {:<8.6f}'.format('Computational complexity: ', macsT))
# print('{:<30}  {:<8.6f}'.format('Number of parameters: ', paramsT))