import torch.nn as nn
import torch
import torch.nn.functional as F

class SobelEdgeDetection(nn.Module):
    def __init__(self):
        super(SobelEdgeDetection, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge

class EdgeAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttentionModule, self).__init__()
        self.edge_detector = SobelEdgeDetection()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        gray_x = torch.mean(x, dim=1, keepdim=True)

        edge = self.edge_detector(gray_x)
        edge = edge.expand(-1, self.channels, -1, -1).mul(x)

        edge_attention = self.conv1(edge)
        edge_attention = self.sigmoid(edge_attention)
        x = x * edge_attention
        x = self.conv2(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        avg_pool = self.fc2(self.relu(self.fc1(avg_pool)))
        return (x * self.sigmoid(avg_pool)) + x

class MutiScaleLayerAttention(nn.Module):
    def __init__(self, channels, depth_size):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.convR0 = nn.Conv3d(channels, channels, kernel_size=1)
        self.convR1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.convR2 = nn.Conv3d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.convR3 = nn.Conv3d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.planePooling = nn.AdaptiveAvgPool3d((depth_size, 1, 1))
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels)

        # Adding SEBlock for Channel Attention
        self.se_block = SEBlock(channels)

    def forward(self, x):
        x1 = self.relu(self.gn(self.convR0(x)))
        x2 = self.relu(self.gn(self.convR1(x)))
        x3 = self.relu(self.gn(self.convR2(x)))
        x4 = self.relu(self.gn(self.convR3(x)))

        # Multi-scale feature fusion
        x = x1 + x2 + x3 + x4

        x = self.relu(self.gn(self.conv1(x)))

        x0 = torch.tanh(self.relu(self.gn(self.conv2(self.planePooling(x)))))
        x = x + x * x0

        # Apply SEBlock for Channel Attention
        x = self.se_block(x)

        return x

class feature_map(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(feature_map, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=int(outchannel / 2), num_channels=outchannel),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1))
        )
        self.conv12 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=int(outchannel / 2), num_channels=outchannel),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1))
        )
        self.channel = outchannel
        self.convtz = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=[2, 1, 1], bias=False),
            nn.GroupNorm(num_groups=int(outchannel / 2), num_channels=outchannel),
        )
        self.attention11 = MutiScaleLayerAttention(32, 128)
        self.attention12 = MutiScaleLayerAttention(32, 32)
        self.attention13 = MutiScaleLayerAttention(32, 8)
        self.attention15 = MutiScaleLayerAttention(64, 8)


    def forward(self, x):
        if x.size()[1] == 32:
            x = self.conv12(x)
            x = F.max_pool3d(x, kernel_size=(2, 1, 1))
            x = self.attention12(x)
            x = self.conv12(x)
            x = F.max_pool3d(x, kernel_size=(2, 1, 1))
            x = self.attention13(x)
            x = self.conv11(x)
            x = F.max_pool3d(x, kernel_size=(4, 1, 1))
        if x.size()[1] == 64:
            x = self.conv12(x)
            x = F.max_pool3d(x, kernel_size=(2, 1, 1))
            x = self.attention15(x)
            x = self.conv12(x)
            x = F.max_pool3d(x, kernel_size=(4, 1, 1))
        if x.size()[1] == 128:
            x = self.conv12(x)
            x = F.max_pool3d(x, kernel_size=(4, 1, 1))
        if x.size()[1] == 256:
            x = self.conv12(x)
        out3 = x.squeeze(dim=2)

        return out3

class conv_block2d(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(conv_block2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.GroupNorm(num_groups=int(outchannel / 2), num_channels=outchannel),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.3),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.GroupNorm(num_groups=int(outchannel / 2), num_channels=outchannel),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.3),
        )
        self.layer1 = nn.Conv2d(inchannel, outchannel, 1, 1, 0, padding_mode='reflect', bias=False)

    def forward(self, x):
        out = self.layer(x)
        x = self.layer1(x)
        out = torch.add(out, x)
        return out

class conv_block(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=1):
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=int(outchannel / 2), num_channels=outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=int(outchannel / 2), num_channels=outchannel),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer(x)
        x = self.layer1(x)
        out = torch.add(out, x)
        return out

class Up(nn.Module):
    def __init__(self, channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels // 2, channels // 2, kernel_size=2, stride=2)
        self.conv = conv_block2d(channels, channels // 2)
        self.conv11 = conv_block2d(channels, channels // 2)

    def forward(self, x1, x2):
        x1 = self.conv11(x1)
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        # x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                                   diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv2d(nn.Module):
    def __init__(self, channels, n_class):
        super(OutConv2d, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(channels // 2, n_class, kernel_size=1)

    def forward(self, x):
        return self.conv(self.conv1(x))


class unet_3(nn.Module):
    def __init__(self):
        super(unet_3, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=16, num_channels=32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=(4, 3, 3), stride=[2, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=16, num_channels=32),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=(4, 3, 3), stride=[2, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=64, num_channels=128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size=(4, 3, 3), stride=[2, 1, 1], padding=[1, 1, 1]),
            nn.GroupNorm(num_groups=64, num_channels=128),
            nn.ReLU(inplace=True),
        )

        self.attention1 = MutiScaleLayerAttention(32, 128)
        self.attention2 = MutiScaleLayerAttention(64, 32)
        self.attention3 = MutiScaleLayerAttention(128, 8)

        self.conv0 = conv_block(1, 32)
        self.conv1 = conv_block(32, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)

        self.feature_conv1 = feature_map(32, 32)
        self.feature_conv2 = feature_map(64, 64)
        self.feature_conv3 = feature_map(128, 128)
        self.feature_conv4 = feature_map(256, 256)
        self.bam1 = EdgeAttentionModule(32)
        self.bam2 = EdgeAttentionModule(64)
        self.bam3 = EdgeAttentionModule(128)
        self.up1 = Up(256)
        self.up2 = Up(128)
        self.up3 = Up(64)

        self.conv_final = OutConv2d(32, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        x0_01 = self.conv0(x)
        x0_00 = self.attention1(x0_01)
        x1_01 = self.conv11(x0_00)
        x1_00 = F.max_pool3d(x1_01, kernel_size=(2, 1, 1))
        x1_02 = self.conv1(x1_00)
        x1_00 = self.attention2(x1_02)
        x2_01 = self.conv12(x1_00)
        x2_00 = F.max_pool3d(x2_01, kernel_size=(2, 1, 1))
        x2_03 = self.conv2(x2_00)
        x2_00 = self.attention3(x2_03)
        x3_01 = self.conv13(x2_00)
        x3_00 = F.max_pool3d(x3_01, kernel_size=(4, 1, 1))
        enc1 = self.feature_conv1(x0_00)
        enc2 = self.feature_conv2(x1_00)
        enc3 = self.feature_conv3(x2_00)
        x3_0 = self.conv3(x3_00)
        x3_0 = x3_0.squeeze(dim=2)
        x1 = self.bam1(enc1)
        x2 = self.bam2(enc2)
        x3 = self.bam3(enc3)
        out = self.up1(x3_0, x3)
        out = self.up2(out, x2)
        out = self.up3(out, x1)
        out = self.conv_final(out)
        out = self.Th(out)

        return out

