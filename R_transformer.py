import torch
import torch.nn as nn

from timm.models import register_model
from timm.models.vision_transformer import _cfg

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.wq(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.depth ** 0.5
        attention = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(x + attention)
        forward = self.feed_forward(x)
        return self.norm2(x + forward)

class RTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x, dim=1)
        return self.decoder(x)

# Model instantiation example
@register_model
def RT(**kwargs):
    model = RTransformer(input_dim=2048, d_model=64, num_heads=2, num_layers=2,num_classes=26)
    return model
