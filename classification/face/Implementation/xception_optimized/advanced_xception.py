import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# CBAM: Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        attention_map = self.spatial_attention(x)
        x = x * attention_map
        return x, attention_map  # Return both output and attention map


# SE Block: Squeeze-and-Excitation
class SEBlock(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_planes, in_planes // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_planes // reduction, in_planes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention[:, :, 0].view(batchsize, 1, width, height)  # Return output and attention map


# DropBlock
class DropBlock(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = torch.ones_like(x).bernoulli_(1 - gamma)
        mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        return x * mask * (mask.numel() / mask.sum())


# Improved Xception Model
class ImprovedXception(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ImprovedXception, self).__init__()
        self.base_model = timm.create_model('xception', pretrained=True)

        self.entry_flow = nn.ModuleList()
        self.middle_flow = nn.ModuleList()
        self.exit_flow = nn.ModuleList()

        self.entry_flow.append(nn.Sequential(
            self.base_model.conv1,
            self.base_model.bn1,
            self.base_model.act1,
            self.base_model.conv2,
            self.base_model.bn2,
            self.base_model.act2,
            SEBlock(128)
        ))
        self.entry_flow.append(nn.Sequential(
            self.base_model.block1,
            SEBlock(256)
        ))
        self.entry_flow.append(nn.Sequential(
            self.base_model.block2,
            SEBlock(728)
        ))

        for i in range(8):
            self.middle_flow.append(nn.Sequential(
                self.base_model.__getattr__(f'block{i + 3}'),
                CBAM(728)
            ))

        self.exit_flow.append(nn.Sequential(
            self.base_model.block11,
            SEBlock(1024)
        ))
        self.exit_flow.append(nn.Sequential(
            self.base_model.conv3,
            self.base_model.bn3,
            self.base_model.act3,
            self.base_model.conv4,
            self.base_model.bn4,
            self.base_model.act4,
            SEBlock(2048)
        ))

        self.global_pool = self.base_model.global_pool
        self.drop_block = DropBlock(drop_prob=0.1, block_size=7)
        self.self_attention = SelfAttention(2048)
        self.fc = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier_name = 'fc'

    def forward(self, x):
        for block in self.entry_flow:
            x = block(x)

        for block in self.middle_flow:
            x, _ = block(x)  # Ignore attention map in forward pass

        for block in self.exit_flow:
            x = block(x)

        x = self.global_pool(x)
        x = self.drop_block(x)
        x, _ = self.self_attention(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_attention_maps(self, x):
        cbam_attention = None
        self_attention_map = None

        for block in self.entry_flow:
            x = block(x)

        for block in self.middle_flow:
            x, att_map = block(x)
            if cbam_attention is None:
                cbam_attention = att_map  # Use last CBAM attention map

        for block in self.exit_flow:
            x = block(x)

        x = self.global_pool(x)
        x = self.drop_block(x)
        x, self_attention_map = self.self_attention(x)

        return {'cbam': cbam_attention, 'self_attention': self_attention_map}