import torch
import torch.nn as nn
from timm.models.layers import DropPath,  trunc_normal_
import math
from model_blocks import TransformerBlock
from dynunet_block import ConvBlock, UnetBasicBlock, UnetOutBlock
import torch.nn.functional as F

NORM_EPS = 1e-5


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x = self.fc1(x_s)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)

        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x = self.fc2(x_s)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., shift_size=4,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, shift_size=shift_size)
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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        )

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """foward function"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossLayer(nn.Module):
    def __init__(self):
        super(CrossLayer, self).__init__()

    def forward(self, a, b, c):
        mid = F.softmax(b @ c.transpose(-2, -1), dim=-1)
        out = a @ mid.transpose(-2,-1)
        out += a
        return out


class TransConvMFF(nn.Module):
    def __init__(self, spatial,in_channels,out_chan,num_heads,kernel_size,stride,padding,encoder,down_up=None):
        super().__init__()
        self.encoder = encoder
        self.res = UnetBasicBlock(spatial_dims=spatial, in_channels=in_channels, out_channels=in_channels,
                       kernel_size=3, stride=1, norm_name='BATCH')
        self.trans = TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=1, bias=False,
                         LayerNorm_type='WithBias')
        self.mff = shiftedBlock(
            dim=in_channels, mlp_ratio=1, drop=0., drop_path=0, norm_layer=nn.BatchNorm2d, shift_size=5)
        if down_up is not None:
            self.down_up = nn.Sequential(
                down_up(in_channels=in_channels, out_channels=out_chan, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_chan, eps=NORM_EPS),
            )
        else :
            self.down_up = None

    def forward(self, x):
        x1 = self.res(x)
        x2 = self.trans(x)
        down = x1 + x2
        y1 = self.mff(down)
        y2 = y1
        if self.down_up:
            y1 = self.down_up(y1)
        if self.encoder:
            return y1, y2
        return y1


class UpMFF(nn.Module):
    def __init__(self, spatial,in_channels,out_chan,num_heads,kernel_size,stride,padding,down_up=None):
        super().__init__()
        self.res = UnetBasicBlock(spatial_dims=spatial, in_channels=out_chan, out_channels=out_chan,
                       kernel_size=3, stride=1, norm_name='BATCH')
        self.trans = TransformerBlock(dim=out_chan, num_heads=num_heads, ffn_expansion_factor=1, bias=False,
                         LayerNorm_type='WithBias')
        self.mff = shiftedBlock(
            dim=out_chan, mlp_ratio=1, drop=0., drop_path=0, norm_layer=nn.BatchNorm2d, shift_size=5)
        if down_up is not None:
            self.down_up = nn.Sequential(
                down_up(in_channels=in_channels, out_channels=out_chan, kernel_size=kernel_size, stride=stride, padding=padding),
            )
        else :
            self.down_up = None

    def forward(self, x,y):
        x = self.down_up(x)
        x = x + y
        x1 = self.res(x)
        x2 = self.trans(x)
        down = x1 + x2
        y1 = self.mff(down)
        return y1


class UpBlock(nn.Module):

    def __init__(self, spatial_dims, upkernsize, upstride, in_channels, out_channels,kernel_size=3, stride=2,
                  norm_name='BATCH', act_name='LEAKYRELU'):
        super().__init__()
        self.transp_conv = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=upkernsize, stride=upstride)
        self.conv_block = ConvBlock(spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, norm_name=norm_name, act_name=act_name, dropout=0)

    def forward(self, x, y):
        out = torch.cat([x,y], dim=1)
        out = self.transp_conv(out)
        out = self.conv_block(out)
        return out


class HmsUnet(nn.Module):
    def __init__(self, in_chans=3, spatial=2, num_classes=1000, embed_dim=32,heads=[2, 2, 4, 8],depth=4):
        super(HmsUnet, self).__init__()
        self.do_ds = False
        self.num_classes = num_classes
        self.conv_op = nn.Conv2d
        self.depth = depth
        self.stem0 = UnetBasicBlock(spatial_dims=spatial,in_channels=in_chans, out_channels=embed_dim,
                                         kernel_size = 3,stride = 2, norm_name='BATCH')
        self.stem1 = UnetBasicBlock(spatial_dims=spatial,in_channels=embed_dim, out_channels=embed_dim*2,
                                         kernel_size = 3,stride = 2, norm_name='BATCH')
        self.down_layer1 = TransConvMFF(spatial=spatial,in_channels=embed_dim*2,out_chan=embed_dim*4,num_heads=heads[0],
                                        kernel_size=3,stride=2,padding=1,encoder=True,down_up=nn.Conv2d)
        self.down_layer2 = TransConvMFF(spatial=spatial,in_channels=embed_dim*4,out_chan=embed_dim*8,num_heads=heads[1],
                                        kernel_size=3,stride=2,padding=1,encoder=True,down_up=nn.Conv2d)
        self.down_layer3 = TransConvMFF(spatial=spatial, in_channels=embed_dim * 8, out_chan=embed_dim * 10,num_heads=heads[2],
                                        kernel_size=3, stride=2, padding=1, encoder=True,
                                        down_up=nn.Conv2d)
        self.down_layer4 = TransConvMFF(spatial=spatial, in_channels=embed_dim * 10, out_chan=embed_dim * 10,num_heads=heads[3],
                                        kernel_size=3, stride=2, padding=1, encoder=False)


        self.Layer4up =  UpMFF(spatial=spatial, in_channels=embed_dim * 10, out_chan=embed_dim * 8,num_heads=heads[3],
                                        kernel_size=2, stride=2, padding=0, down_up=nn.ConvTranspose2d)
        self.Layer3up = UpMFF(spatial=spatial, in_channels=embed_dim * 8, out_chan=embed_dim * 4, num_heads=heads[3],
                              kernel_size=2, stride=2, padding=0, down_up=nn.ConvTranspose2d)
        self.Layer2up = UpMFF(spatial=spatial, in_channels=embed_dim * 4, out_chan=embed_dim * 2, num_heads=heads[3],
                              kernel_size=2, stride=2, padding=0, down_up=nn.ConvTranspose2d)
        self.Layer1up = UpBlock(spatial_dims=spatial, in_channels=embed_dim*4, out_channels=embed_dim*2, upkernsize=2, upstride=2,
                                kernel_size = 3, stride = 1,
                                 drop = 0, norm_name = 'BATCH', act_name = 'LEAKYRELU')
        self.Layer0up = UpBlock(spatial_dims=spatial, in_channels=embed_dim*3, out_channels=embed_dim*1, upkernsize=2, upstride=2,
                                kernel_size = 3, stride = 1,
                                 drop = 0, norm_name = 'BATCH', act_name = 'LEAKYRELU')


        self.deep1 = UnetOutBlock(spatial_dims=spatial, in_channels=embed_dim, out_channels=num_classes)
        self.deep2 = UnetOutBlock(spatial_dims=spatial, in_channels=embed_dim*2, out_channels=num_classes)
        self.deep3 = UnetOutBlock(spatial_dims=spatial, in_channels=embed_dim*2, out_channels=num_classes)
        self.deep4 = UnetOutBlock(spatial_dims=spatial, in_channels=embed_dim*4, out_channels=num_classes)
        self.deep5 = UnetOutBlock(spatial_dims=spatial, in_channels=embed_dim*8, out_channels=num_classes)

        self.adown2 = nn.Sequential(nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=3, stride=2, padding=1, groups=embed_dim),
                                    nn.Conv2d(embed_dim*4, embed_dim*8, kernel_size=3, stride=2, padding=1, groups=embed_dim),
                                    )
        self.bdown1 = nn.Conv2d(embed_dim*4, embed_dim*8, kernel_size=3, stride=2, padding=1, groups=embed_dim)
        self.cross3 = CrossLayer()

        self.sigmoid = nn.Sigmoid()
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(
                m, nn.ConvTranspose2d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        list_down = []
        x = self.stem0(x)
        list_down.append(x)
        x = self.stem1(x)
        list_down.append(x)
        x,y = self.down_layer1(x)
        list_down.append(y)
        x, y = self.down_layer2(x)
        list_down.append(y)
        x, y = self.down_layer3(x)
        list_down.append(y)
        x = self.down_layer4(x)
        adown2 = self.adown2(list_down[2])
        bdown1 = self.bdown1(list_down[3])
        down3 = self.cross3(list_down[4], adown2, bdown1)
        up4 = self.Layer4up(x, list_down[-1]+down3)
        up3 = self.Layer3up(up4, list_down[-2])
        up2 = self.Layer2up(up3, list_down[-3])
        up1 = self.Layer1up(up2, list_down[-4])
        up0 = self.Layer0up(up1, list_down[-5])

        out0 = self.deep1(up0)
        out0 = self.sigmoid(out0)
        if self.do_ds:
            out1 = self.deep2(up1)
            out2 = self.deep3(up2)
            out3 = self.deep4(up3)
            return [out0, out1, out2, out3]
        else:
            return out0


