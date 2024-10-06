import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import math
from torchsummary import summary
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg

class DDConv(nn.Module):
    def __init__(self, dim):
        super(DDConv, self).__init__()
        self.ddconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        x = self.ddconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.ddconv = DDConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ddconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DDCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv1d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DDCA(d_model)
        self.proj_2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        norm1_x = self.norm1(x)
        # Step 2: Pass the normalized input through the attention layer
        attn_x = self.attn(norm1_x)
        # Step 3: Scale the output of the attention layer
        scaled_attn_x = self.layer_scale_1.unsqueeze(-1) * attn_x
        # Step 4: Apply DropPath to the scaled output
        dropped_scaled_attn_x = self.drop_path(scaled_attn_x)
        # Step 5: Add the result to the original input (Residual Connection)
        x = x + dropped_scaled_attn_x
        # Repeat the above steps for the second part
        # Step 6: Normalize the input
        norm2_x = self.norm2(x)
        # Step 7: Pass the normalized input through the MLP layer
        mlp_x = self.mlp(norm2_x)
        # Step 8: Scale the output of the MLP layer
        scaled_mlp_x = self.layer_scale_2.unsqueeze(-1) * mlp_x
        # Step 9: Apply DropPath to the scaled output
        dropped_scaled_mlp_x = self.drop_path(scaled_mlp_x)
        # Step 10: Add the result to the input (Residual Connection)
        x = x + dropped_scaled_mlp_x
        return x


class OverlapPatchEmbed(nn.Module):
    """ Signal to Patch Embedding
    """
    def __init__(self, signal_length=1024, patch_size=7, stride=4, in_chans=2, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x



class WACN(nn.Module):
    def __init__(self, signal_length=1024, in_chans=2, num_classes=24, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0.3, drop_path_rate=0.3, norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=15, padding=7),  # Adjusted for 2 input channels
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, padding=7),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(signal_length=signal_length if i == 0 else signal_length // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=2 if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x = self.cnn_layers(x)
        for i in range(self.num_stages):
            # print('num_stages={}'.format(i))
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            #print('x.shape={},L={}'.format(x.shape, L))
            for blk in block:
                x = blk(x)
            #print('x.shape={},after block'.format(x.shape))
            x = x.transpose(1, 2)
            #print('x.shape={},after flatten and transpose'.format(x.shape))
            x = norm(x)
           # print('x.shape={},after norm'.format(x.shape))
            if i != self.num_stages - 1:


                x = x.permute(0, 2, 1)
               # print('x.shape={},after permute'.format(x.shape))
                x = x.contiguous()
              #  print('x.shape={},after contiguous'.format(x.shape))
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)

        x = self.head(x)
        return x

@register_model
def wacn_b0(num_classes=24, **kwargs):
    model = WACN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        num_classes=num_classes, **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def wacn_b1(num_classes=24, **kwargs):
    model = WACN(num_classes=num_classes,
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def wacn_b2(**kwargs):
    model = WACN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def wacn_b3(**kwargs):
    model = WACN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    model.default_cfg = _cfg()

    return model


def compute_flops(model, input_shape):
    # 输入数据形状为 input_shape，元素值为随机数
    input_data = torch.randn((1, ) + input_shape).cuda()

    # 修改此处，使之能够接受 (dim, 1024) 形状的输入数据
    # 我们需要将 input_shape 作为一个元组传递给 get_model_complexity_info 函数
    macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                             print_per_layer_stat=True, verbose=True)
    return macs,params



dim = 2  # 根据你的实际输入数据的通道数设置
MSCA = wacn_b0().cuda()

macsT, paramsT = compute_flops(MSCA, (dim, 1024))


print('{:<30}  {:<8.6f}'.format('Computational complexity: ', macsT))
print('{:<30}  {:<8.6f}'.format('Number of parameters: ', paramsT))



# model = van_b0(num_classes=11)
# # # 打印模型信息
# network_name = type(model).__name__
# print("Network name:", network_name)
# # # 打印模型结构
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# summary(model, (2, 1024))
