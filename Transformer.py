import torch
import numpy as np
import torch.nn as nn
import math
from torchsummary import summary


def preprocess_data(data, N, M, P, Q):
    batch_size = data.shape[0]

    Len = data.shape[2]
    # 检查 N 和 M 的值是否满足条件
    assert 2 * Len == N * M, "2Len should be equal to N*M"
    assert Len % M == 0
    result = []
    for batch_idx in range(batch_size):
        arr1 = data[batch_idx, 0, :].cpu().numpy()
        arr2 = data[batch_idx, 1, :].cpu().numpy()
        arr1 = arr1.reshape(N // 2, M)
        arr2 = arr2.reshape(N // 2, M)

        reshaped_arr = np.zeros((N, M))
        row_index = 0

        for i in range(N):
            if i % 2 == 0:
                reshaped_arr[i] = arr1[row_index]
            else:
                reshaped_arr[i] = arr2[row_index]
                row_index += 1
        # 将 x 转换成一个 N×M 的矩阵
        R = reshaped_arr

        # 计算子矩阵的数量
        Z1, Z2 = N // P, M // Q

        # 将 R 分割为多个 P×Q 的子矩阵
        R_patches = []
        for z1 in range(Z1):
            for z2 in range(Z2):
                patch = R[z1 * P: (z1 + 1) * P, z2 * Q: (z2 + 1) * Q]
                patch = patch.reshape(-1)  # (4)
                R_patches.append(patch)

        # 将所有子矩阵添加到结果列表中
        result.append(R_patches)
    # 在返回结果之前，将其转换回 PyTorch 张量并移动到原始设备上
    result = np.array(result)  # Add this line to convert list to numpy array
    return torch.from_numpy(result.astype(np.float32)).to(data.device)


class DataPreprocessor(nn.Module):
    def __init__(self, N, M, P, Q):
        super(DataPreprocessor, self).__init__()
        self.N = N
        self.M = M
        self.P = P
        self.Q = Q

    def forward(self, data):
        return preprocess_data(data, self.N, self.M, self.P, self.Q)


class LinearProjectionLayer(nn.Module):
    def __init__(self, PQ, D, Z1, Z2):
        super(LinearProjectionLayer, self).__init__()
        self.PQ = PQ
        self.D = D
        self.Z1 = Z1
        self.Z2 = Z2
        self.embedding = nn.Linear(PQ, D)
        self.class_token = nn.Parameter(torch.randn(1, 1, D))
        self.position_encoding = nn.Parameter(torch.randn(Z1 * Z2 + 1, D))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.PQ)  # [batch_size, Z1*Z2, PQ]

        # Linear projection
        R_dots = self.embedding(x)  # [batch_size, Z1*Z2, D]

        # Add class token
        c = self.class_token.repeat(batch_size, 1, 1)
        R_hat = torch.cat((c, R_dots), dim=1)  # [batch_size, Z1*Z2 + 1, D]

        # Add position encoding
        R = R_hat + self.position_encoding
        return R


class LN_MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(LN_MHA, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_0 = nn.Linear(d_model, d_model)
        self.LN = nn.LayerNorm(d_model)  # Layer Normalization
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        batch_size = x.size(0)


        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = torch.softmax(attention, dim=-1)

        head = torch.matmul(attention, V)
        head = head.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_0(head)
        output = self.LN(output)
        output = self.dropout(output)
        output = x + output
        return output


class LN_MLP(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(LN_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        ln = self.mlp(x)
        output = self.LN(ln)
        output = self.dropout(output)
        return x + output


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_hidden, num_heads):
        super(TransformerEncoder, self).__init__()
        self.mha = LN_MHA(d_model, num_heads)
        self.mlp = LN_MLP(d_model, d_hidden)
    def forward(self, x):
        x = self.mha(x)
        x = self.mlp(x)
        return x


class TransformerNetwork(nn.Module):
    def __init__(self, N, M, P, Q, D, num_classes, num_heads, num_layers):
        super(TransformerNetwork, self).__init__()
        Z1, Z2 = N // P, M // Q
        d_hidden = 2 * D  # or any number based on your preference

        self.preprocess = DataPreprocessor(N, M, P, Q)
        self.linear_projection = LinearProjectionLayer(P * Q, D, Z1, Z2)

        # now TransformerEncoder is used instead of MultiHeadAttention
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(D, d_hidden, num_heads) for _ in range(num_layers)])

        # Classifier remains the same
        self.classifier = nn.Linear(D, num_classes)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.linear_projection(x)

        for transformer_encoder in self.transformer_encoders:
            x = transformer_encoder(x)

        x = x[:, 0, :]
        x = self.classifier(x)

        return x


def transformer_net(N=64, M=32, P=4, Q=4, D=256, num_classes=24, num_heads=16, num_layers=5):
    # 实例化网络
    network = TransformerNetwork(N, M, P, Q, D, num_classes, num_heads, num_layers)
    return network

"""
model = transformer_net(num_classes=24)
# # 打印模型信息
network_name = type(model).__name__
print("Network name:", network_name)
# # 打印模型结构
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model, (2, 128))
"""