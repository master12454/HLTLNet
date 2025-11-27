import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class FFParser(nn.Module):
    def __init__(self, dim, h=256, w=129):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, spatial_size=None):
        r = x
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size
        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)
        out = x + r

        return out

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

#低秩分解
class GetLS_Net(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_Net, self).__init__()
        self.n = n
        self.num_block = num_block
        self.conv_W00 = ConvLayer(channel, 2*n, s, stride)
        self.lamj = nn.Parameter(torch.rand(1, self.n*2))  # l1-norm
        self.lamz = nn.Parameter(torch.rand(1, 1))
        for i in range(num_block):
            self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))
    def _make_layer(self, input_channels,  output_channels, num_blocks=1):
        block = Res_CBAM_block
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        b, c, h, w = x.shape
        tensor_l = self.conv_W00(x)  # Z_0
        tensor_z = eta_l1(tensor_l, self.lamj)
        S_history = []
        for i in range(self.num_block):
            lrrblock = getattr(self, 'lrrblock' + str(i))
            tensor_z = lrrblock(x, tensor_z, self.lamj, self.lamz)
            # 提取当前稀疏分量并保存
            S_current = tensor_z[:, self.n: 2 * self.n, :, :]
            S_history.append(S_current)
        L = tensor_z[:, :self.n//2, :, :]  # 前n/2个通道 -> L
        D = tensor_z[:, self.n//2: 1 * self.n, :, :]  # n/2到n个通道 -> D
        S = tensor_z[:, self.n: 2 * self.n, :, :]  # 后n个通道 -> S
        return L, S, D, S_history

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class LRR_Block_lista(nn.Module):
    def __init__(self, s, n, c, stride):
        super(LRR_Block_lista, self).__init__()
        self.conv1_Wdz= self._make_layer(n, c, 2)
        self.conv1_Wdtz = self._make_layer(c, n, 2)
        self.attenton_1 = ECA(48)
    def _make_layer(self, input_channels,  output_channels, num_blocks=1):
        block = Res_CBAM_block
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    def forward(self, x, tensor_z, lam_theta, lam_z):
        convZ1 = self.conv1_Wdz(tensor_z)
        midZ = x - convZ1
        tensor_c = lam_z*tensor_z + self.conv1_Wdtz(midZ)
        tensor_c = self.attenton_1(tensor_c)
        Z = eta_l1(tensor_c, lam_theta)
        return Z

def eta_l1(r_, lam_):
    # l_1 norm based
    # implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)
    B, C, H, W = r_.shape
    lam_ = torch.reshape(lam_, [1, C, 1, 1])
    lam_ = lam_.repeat(B, 1, H, W)
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam_, 0)
    return R


class LTLNET(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, dataset_name=None, mode='test'):
        super(LTLNET, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.conv0 = self._make_layer(block, input_channels, 16)
        self.conv1 = self._make_layer(block, 16, 32, 2)
        self.conv2 = self._make_layer(block, 128, 16)
        self.convR = self._make_layer(block, 32, 16)
        if dataset_name == 'NUDT':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 5)
            self.new_fusion = new_fusion(128, 16)
            # self.new_fusion = new_fusion(112, 16)
        elif dataset_name == 'NUAA':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 4)
            self.new_fusion = new_fusion(112, 16)
        elif dataset_name == 'IRSTD':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 6)
            self.new_fusion = new_fusion(144, 16)
        else:
            self.get_ls = GetLS_Net(3, 16, 16, 1, 5)
            self.new_fusion = new_fusion(128, 16)
        self.FFP = FFParser(16, 256, 129)
        self.attenton_0 = ECA(16)
        self.attenton_1 = ECA(32)
        self.final_z = nn.Conv2d(16, num_classes, kernel_size=1)
        self.final_d = nn.Conv2d(8, num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0 = self.conv0(input)
        L, S, D, S_history = self.get_ls(x0)
        FS = self.FFP(S)
        FS_history = [self.FFP(s) for s in S_history]

        x1 = self.conv1(self.pool(self.attenton_0(FS)))
        z1 = self.attenton_1(x1)
        z2 = self.up(z1)
        z3 = self.convR(z2)
        z0 = torch.cat([*FS_history, FS, z2], 1)
        z0 = self.new_fusion(z0,z3)

        if self.mode == 'train':
            output = self.final_z(z0).sigmoid()
            output_S = self.final_z(FS).sigmoid()
            output_L = self.final_d(L).sigmoid()
            output_D = self.final_d(D).sigmoid()
            out_FS_history = [self.final_z(fs).sigmoid() for fs in FS_history]
            return output, output_L, output_S, output_D, out_FS_history
        elif self.mode == 'test':
            output = self.final_z(z0).sigmoid()
            return output

class PAFM(nn.Module):
    def __init__(self, channels_high, out_channels ):
        super(PAFM, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_channels=channels_high, out_channels=channels_high, kernel_size=3, stride=2, padding=1)
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_high, channels_high // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_high)
        )
        self.AAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels_high, channels_high // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_high)
        )
        self.active = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_high, channels_high // 4, 1),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, 1),
            nn.BatchNorm2d(channels_high),
            nn.Sigmoid()

        )
    def forward(self, x_high,x):
        _, _, h, w = x_high.shape
        x_sum = x_high +  x
        x_conv = self.conv(x_sum)
        p_x_high = self.active((self.GAP(x_high) + self.AAP(x_high)))
        p_x_high =  F.interpolate(p_x_high, scale_factor=h // 4, mode='nearest')
        output = ((x_conv) * x_sum) * p_x_high

        return output
class new_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(new_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        self.fuse = PAFM(16,16)
    def forward(self, x,r):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.fuse(x,r)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += x
        out = self.relu(out)
        return out
