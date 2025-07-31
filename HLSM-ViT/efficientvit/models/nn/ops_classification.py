# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast

from efficientvit.models.nn.act import build_act
from efficientvit.models.nn.norm import build_norm
from efficientvit.models.utils import get_same_padding, list_sum, resize, val2list, val2tuple
# from my_some_test import get_list_dimensions
__all__ = [
    "ConvLayer",
    "UpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "EfficientViTBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        # print('头的数量是：', heads,  'dim是：', self.dim, '输入形状是：', in_channels)
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
        # self.proj1 = ConvLayer(
        #     total_dim,
        #     out_channels,
        #     1,
        #     use_bias=use_bias[1],
        #     norm=norm[1],
        #     act_func=act_func[1],
        # )
        #加权融合部分代码
        self.global_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        self.weight_generator = nn.Linear(in_channels*2, 3)
        self.weight_generator_outuse = nn.Linear(in_channels, 1)#专给外部通道注意力融合用的权重
        # self.weight_generator = nn.Sequential(
        #     nn.Linear(2*in_channels, 2*in_channels //2),
        #     nn.ReLU(),
        #     nn.Linear(2*in_channels // 2, 3)
        # )
        self.sigmoid = nn.Sigmoid()
        # ###多尺度用这个dwc，非多尺度用下面那个dwc
        self.dwc = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=5,
                             groups=in_channels*2, padding=5 // 2)
        # self.dwc = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5,
        #                      groups=in_channels, padding=5 // 2)
        # self.scale = nn.Parameter(torch.zeros(size=(1, 1, in_channels*2, 1)))
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, 1, dim)))##faltten要用到的scale
        self.scale_add = nn.Parameter(torch.ones(1))#加到最终结果上的时候也可以定义一个自动学习的参数
        ##############################作用在输入x上的卷积
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def select_important_tokens(self, tensor, topk):
        """
        根据均值和标准差选择最重要的token。
        """
        # 计算每个token的均值和标准差
        mean = tensor.mean(dim=-1)
        std = tensor.std(dim=-1)
        # batch_size, seq_length, d_model = tensor.size()
        # 计算重要性得分
        importance_score = mean + std
        
        # 选择得分最高的topk个token 
        _, top_indices = torch.topk(importance_score, topk, dim=2)
        """
        根据L2范数选择最重要的token。
        """
        # # 计算每个token在最后一个维度上的L2范数
        # norm_scores = torch.norm(tensor, p=2, dim=-1)
        
        # # 选择得分最高的topk个token
        # _, top_indices = torch.topk(norm_scores, topk, dim=-1)
        """
        根据最大激活值选择最重要的token。
        """
        # # 计算每个token在最后一个维度上的最大值
        # activation_scores = tensor.max(dim=-1).values
    
        # # 选择得分最高的topk个token
        # _, top_indices = torch.topk(activation_scores, topk, dim=-1)
        return top_indices
    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        # print(qkv.shape)
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )
        # print(q.shape)
        # print(B, H, W)  #128 8 8 或 128 4 4 
        # print('q的形状是：', q.shape)#[128, 16, 64, 16]或[128, 32, 16, 16]或[128, 16, 196, 16]或[128, 32, 49, 16]
        #所以q的形状中，第2维是h*w，第1维是按需制定，第三维是16固定
        # lightweight linear attention
        #在这加上采样softmax注意力模块###########################################
        sample_num = q.shape[2]
        # sample_idx = torch.linspace(0, sample_num-1, steps=int(2 * math.sqrt(sample_num))).long()  # 均匀采样一些点
        sample_idx = self.select_important_tokens(q, int(2 * math.sqrt(sample_num)))
        ####改进版采样
        # 扩展 topk_indices 以匹配 q 的形状
        expanded_indices = sample_idx.unsqueeze(-1).expand(-1, -1, -1, q.size(-1))
        # 使用 gather 获取 sampled_q，形状为 [128, 16, 8, 16]
        q_sampled = torch.gather(q, 2, expanded_indices)
        k_sampled = torch.gather(k, 2, expanded_indices)
        v_sampled = torch.gather(v, 2, expanded_indices)
        ####改进版采样结束，主要替换qkvsampled
        d_k = k_sampled.size(-1)
        attention_scores = torch.matmul(q_sampled, k_sampled.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        attention_weights = F.softmax(attention_scores, dim=-1)
        top_percentage = 50
        num_to_keep = int(attention_weights.size(-1) * top_percentage / 100)
        sorted_weights, _ = attention_weights.sort(dim=-1)
        threshold_value = sorted_weights[..., -num_to_keep]
        # 仅保留大于等于阈值的权重，其他设置为0
        attention_weights = torch.where(attention_weights >= threshold_value.unsqueeze(-1), attention_weights, torch.zeros_like(attention_weights))
        sampled_attention = torch.matmul(attention_weights, v_sampled)
        qk_result_expanded = torch.zeros_like(v)
        ##新std的sample方法需要用下面这一行不用下面的for循环
        qk_result_expanded.scatter_(2, expanded_indices, sampled_attention)
        sampled_result = qk_result_expanded.reshape(B, -1, H, W)
        # #可以在这加上聚焦函数和dwc################################################
        v1_reshaped = v.reshape(B, -1, H, W)
        add_featuremap = self.dwc(v1_reshaped)
        # print('我转换的v的特征图的形状是', v1_reshaped.shape)
        #下面是用聚焦函数的代码###############################################################################3
        scale = nn.Softplus()(self.scale)
        q = self.kernel_func(q) + 1e-6
        k = self.kernel_func(k) + 1e-6
        
        #用不用聚焦函数的核心就是这一段
        q = q / scale
        k = k / scale
        # print('经过kernel_func后q的形状是：', q.shape)#不变
        # linear matmul
        q_norm = q.norm(dim=-2, keepdim=True)
        k_norm = k.norm(dim=-2, keepdim=True)
        q = q ** 3
        k = k ** 3
        q = (q / q.norm(dim=-2, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-2, keepdim=True)) * k_norm


        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))

        ###用softmax注意力
        # att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        # original_dtype = att_map.dtype
        # if original_dtype in [torch.float16, torch.bfloat16]:
        #     att_map = att_map.float()
        # att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        # att_map = att_map.to(original_dtype)
        # out = torch.matmul(v, att_map)  # b h d n

        # out = torch.reshape(out, (B, -1, H, W))
        ####################################################聚焦函数结尾
        # print('输出out的形状是', out.shape)
        #特征融合代码
        bn,cn,hn,wn = out.shape 
        pooled_sample = self.global_pool(sampled_result).view(bn, cn)
        pooled_focus = self.global_pool(out).view(bn, cn)
        pooled_dwc = self.global_pool(add_featuremap).view(bn, cn)
        weights1 = self.sigmoid(self.weight_generator(pooled_sample))
        weights2 = self.sigmoid(self.weight_generator(pooled_focus))
        weights3 = self.sigmoid(self.weight_generator(pooled_dwc))
        # print('采样结果形状',pooled_sample.shape)
        # print('dwc结果形状',pooled_dwc.shape)
        # print('focus结果形状',pooled_focus.shape)
        # out = out * weights2[:,1].view(bn,1,1,1) + sampled_result * weights1[:,0].view(bn,1,1,1) + add_featuremap * weights3[:,2].view(bn,1,1,1)
        out = out * weights2[:,1].view(bn,1,1,1) + sampled_result * weights1[:,0].view(bn,1,1,1)
        # out = out * weights2[:,1].view(bn,1,1,1) + add_featuremap * weights3[:,2].view(bn,1,1,1)
        # out = out * weights2[:,1].view(bn,1,1,1) + sampled_result * weights1[:,0].view(bn,1,1,1)
        # out = out + add_featuremap + self.scale_add * sampled_result
        ###############################################原始代码
        # lightweight linear attention
        # q = self.kernel_func(q)
        # k = self.kernel_func(k)

        # # linear matmul
        # trans_k = k.transpose(-1, -2)

        # v = F.pad(v, (0, 1), mode="constant", value=1)
        # kv = torch.matmul(trans_k, v)
        # out = torch.matmul(q, kv)
        # out = out[..., :-1] / (out[..., -1:] + self.eps)

        # out = torch.transpose(out, -1, -2)
        # out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # # generate multi-scale q, k, v
        #########对输入做一下卷积
        result_conv = self.conv3x3(x)
        result_conv = self.conv5x5(result_conv)
        ca = self.channel_attention(result_conv)
        result_conv = result_conv * ca
        #求通道注意力的权重
        bo, co, _, _ = x.shape
        pooled_channel_attention = self.global_pool(result_conv).view(bo,co)
        weights_ca = self.sigmoid(self.weight_generator_outuse(pooled_channel_attention))
        # print('shape', x.shape)
        # print(pooled_channel_attention.shape)
        # print(weights_ca)
        #############x经过两个卷积和一个通道注意力，形状不变
        qkv = self.qkv(x)
        # print('qkv的形状', qkv.shape)#[128, 384, 8, 8]或[128, 768, 4, 4]
        multi_scale_qkv = [qkv]
        # print('初始的qkv形状', multi_scale_qkv.shape)
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        # print('多尺度后的的qkv形状', multi_scale_qkv.shape)
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        # print('拼接后的qkv形状', multi_scale_qkv.shape)#[128, 1536, 4, 4]或[128, 768, 8, 8]
        # print(multi_scale_qkv.shape)
        out = self.relu_linear_att(multi_scale_qkv)
        # print('经过relu_linear_att后的形状：', out.shape)
        # out = self.proj(out)+result_conv
        # out = self.proj(out)
        out = self.proj(out) + result_conv * weights_ca.view(bo, 1, 1, 1)
        #############################不要多尺度，原始的线性注意力
        # # print(qkv.shape)
        # out = self.relu_linear_att(qkv)
        # out = self.proj1(out)
        # # print(out.shape)
        #####################################################
        

        return out

    @staticmethod
    def configure_litemla(model: nn.Module, **kwargs) -> None:
        eps = kwargs.get("eps", None)
        for m in model.modules():
            if isinstance(m, LiteMLA):
                if eps is not None:
                    m.eps = eps


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
    ):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
