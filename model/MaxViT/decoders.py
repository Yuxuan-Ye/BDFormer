import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as  np


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv_out(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv_out,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAttentionOut(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionOut, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        print('1. x.shape=', x.shape)
        print('2. avg_pool_out.shape=', avg_pool_out.shape)
        print('3. fc1.shape=', self.fc1(avg_pool_out).shape)
        print('4. relu.shape=', self.relu1(self.fc1(avg_pool_out)))
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
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

class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Cat,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        self.CA1 = ChannelAttention(2*channels[3])
        
        self.SA = SpatialAttention()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = torch.cat((x3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = torch.cat((x2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = torch.cat((x1,d1),dim=1)
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
        

class CASCADE_Add(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Add,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # aggregate 3
        d3 = d3 + x3
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # aggregate 2
        d2 = d2 + x2
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # aggregate 1
        d1 = d1 + x1
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class Normal_Decoder(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], num_classes=4):
        super(Normal_Decoder, self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])

        self.Up0 = up_conv_out(ch_in=channels[3], ch_out=num_classes)
        # self.AG0 = Attention_block(F_g=num_classes, F_l=num_classes, F_int=int(num_classes / 2))
        self.AG0 = Attention_block(F_g=num_classes, F_l=num_classes, F_int=num_classes)
        self.ConvBlock0 = conv_block(ch_in=num_classes, ch_out=num_classes)

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        self.CA0 = ChannelAttentionOut(num_classes)

        self.SA = SpatialAttention()

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        # upconv3
        d3 = self.Up3(d4)

        # AG3
        x3 = self.AG3(g=d3, x=skips[0])

        # aggregate 3
        d3 = d3 + x3

        # CAM3
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        # upconv2
        d2 = self.Up2(d3)

        # AG2
        x2 = self.AG2(g=d2, x=skips[1])

        # aggregate 2
        d2 = d2 + x2

        # CAM2
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        # print(d2.shape)
        d2 = self.ConvBlock2(d2)

        # upconv1
        d1 = self.Up1(d2)

        # print(skips[2])
        # AG1
        x1 = self.AG1(g=d1, x=skips[2])

        # aggregate 1
        d1 = d1 + x1

        # CAM1
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)

        # upconv0
        d0 = self.Up0(d1)
        # AG0
        x0 = self.AG0(g=d0, x=d0)
        # aggregate 0
        d0 = d0 + x0
        # CAM0
        d0 = self.CA0(d0) * d0
        d0 = self.SA(d0) * d0
        d0 = self.ConvBlock0(d0)
        return d0
        # return d4, d3, d2, d1, d0


'''***** FCT Decoder *****'''
class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=1)  # num_heads=self.num_heads)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1


class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = Attention(channels=out_channels,
                                          num_heads=num_heads,
                                          proj_drop=proj_drop,
                                          padding_q=padding_q,
                                          padding_kv=padding_kv,
                                          stride_kv=stride_kv,
                                          stride_q=stride_q,
                                          attention_bias=attention_bias,
                                          )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3

        # return x


class Wide_Focus(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out


class Block_encoder_bottleneck(nn.Module):
    def __init__(self, blk, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.blk = blk
        if ((self.blk=="first") or (self.blk=="bottleneck")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)
        elif ((self.blk=="second") or (self.blk=="third") or (self.blk=="fourth")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(1, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)


    def forward(self, x, scale_img="none"):
        if ((self.blk=="first") or (self.blk=="bottleneck")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            x1 = F.dropout(x1, 0.3)
            # x1 = F.max_pool2d(x1, (2,2))
            out = self.trans(x1)
            # without skip
        elif ((self.blk=="second") or (self.blk=="third") or (self.blk=="fourth")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2,2))
            out = self.trans(x1)
            # with skip
        return out


# class Block_decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, att_heads, dpr):
#         super().__init__()
#         self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
#         self.upsample = nn.Upsample(scale_factor=2)
#         self.conv1 = nn.Conv2d(2 * in_channels, in_channels, 3, 1, padding="same")
#         self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
#         self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
#         self.trans = Transformer(out_channels, att_heads, dpr)
#
#     def forward(self, x, skip):
#         x1 = x.permute(0, 2, 3, 1)
#         x1 = self.layernorm(x1)
#         x1 = x1.permute(0, 3, 1, 2)
#         x1 = torch.cat((skip, x1), axis=1)
#         x1 = F.relu(self.conv1(x1))
#         x1 = self.upsample(x1)
#         x1 = F.relu(self.conv2(x1))
#         x1 = F.relu(self.conv3(x1))
#         x1 = F.dropout(x1, 0.3)
#         out = self.trans(x1)
#         return out


class Block_decoder_out(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=4)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return x1


class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=4)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        out = torch.sigmoid(self.conv3(x1))

        return out


class FCT_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 2, 2, 2, 1]
        decoder_filters = [768, 384, 192, 96]

        # number of blocks used in the model
        blocks = len(decoder_filters) + 1

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        # shape
        init_sizes = torch.ones((2, 224, 224, 1))
        init_sizes = init_sizes.permute(0, 3, 1, 2)

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # model
        self.bottleneck = Block_encoder_bottleneck("bottleneck", decoder_filters[0], decoder_filters[0], att_heads[0], dpr[0])
        self.Dblock_1 = Block_decoder(decoder_filters[0], decoder_filters[1], att_heads[1], dpr[1])
        self.Dblock_2 = Block_decoder(decoder_filters[1], decoder_filters[2], att_heads[2], dpr[2])
        self.Dblock_3 = Block_decoder(decoder_filters[2], decoder_filters[3], att_heads[3], dpr[3])
        self.Dblock_out = DS_out(decoder_filters[3], 4)

        # self.ds7 = DS_out(decoder_filters[1], 4)
        # self.ds8 = DS_out(decoder_filters[2], 4)
        # self.ds9 = DS_out(decoder_filters[3], 4)

    def forward(self, x, encoder_feature):
        decoder_feature0 = self.bottleneck(x)
        decoder_feature1 = self.Dblock_1(decoder_feature0, encoder_feature[0])
        decoder_feature2 = self.Dblock_2(decoder_feature1, encoder_feature[1])
        decoder_feature3 = self.Dblock_3(decoder_feature2, encoder_feature[2])

        # decoder_feature = [decoder_feature0, decoder_feature1, decoder_feature2, decoder_feature3]
        # return decoder_feature
        x_out = self.Dblock_out(decoder_feature3)
        return x_out


'''***** Multi-task FCT Decoder *****'''
class MTAttention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_kv = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_kv = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=1)  # num_heads=self.num_heads)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "kv":
            x1 = F.relu(self.conv_kv(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_kv(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x, kv):
        q = self._build_projection(x, "q")
        kv = self._build_projection(kv, "kv")

        return q, kv

    def forward(self, x, kv):
        q, kv = self.forward_conv(x, kv)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        kv = kv.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        kv = kv.permute(0, 2, 1)

        x1 = self.attention(query=q, value=kv, key=kv, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1


class MTTransformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = MTAttention(channels=out_channels,
                                          num_heads=num_heads,
                                          proj_drop=proj_drop,
                                          padding_q=padding_q,
                                          padding_kv=padding_kv,
                                          stride_kv=stride_kv,
                                          stride_q=stride_q,
                                          attention_bias=attention_bias,
                                          )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x, kv):
        x1 = self.attention_output(x, kv)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3


class MTBlock_encoder_bottleneck(nn.Module):
    def __init__(self, blk, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.blk = blk
        if ((self.blk=="first") or (self.blk=="bottleneck")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = MTTransformer(out_channels, att_heads, dpr)
        elif ((self.blk=="second") or (self.blk=="third") or (self.blk=="fourth")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(1, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = MTTransformer(out_channels, att_heads, dpr)


    def forward(self, x, kv, scale_img="none"):
        if ((self.blk=="first") or (self.blk=="bottleneck")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            x1 = F.dropout(x1, 0.3)
            # x1 = F.max_pool2d(x1, (2,2))
            out = self.trans(x1, kv)
            # without skip
        elif ((self.blk=="second") or (self.blk=="third") or (self.blk=="fourth")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2,2))
            out = self.trans(x1, kv)
            # with skip
        return out


class MTBlock_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = MTTransformer(out_channels, att_heads, dpr)

    def forward(self, x, kv,skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        # kv = self.upsample(kv)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1, kv)
        return out


class MTFCT_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 2, 2, 2, 1]
        decoder_filters = [768, 384, 192, 96]

        # number of blocks used in the model
        blocks = len(decoder_filters) + 1

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        # shape
        init_sizes = torch.ones((2, 224, 224, 1))
        init_sizes = init_sizes.permute(0, 3, 1, 2)

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # kv conv
        self.kv_conv1 = nn.Conv2d(decoder_filters[0], decoder_filters[1], 1, 1, padding="same")
        self.kv_conv2 = nn.Conv2d(decoder_filters[1], decoder_filters[2], 1, 1, padding="same")
        self.kv_conv3 = nn.Conv2d(decoder_filters[2], decoder_filters[3], 1, 1, padding="same")

        # model SEG
        self.bottleneck1 = MTBlock_encoder_bottleneck("bottleneck", decoder_filters[0], decoder_filters[0], att_heads[0], dpr[0])
        self.Dblock_1_1 = MTBlock_decoder(decoder_filters[0], decoder_filters[1], att_heads[1], dpr[1])
        self.Dblock_1_2 = MTBlock_decoder(decoder_filters[1], decoder_filters[2], att_heads[2], dpr[2])
        self.Dblock_1_3 = MTBlock_decoder(decoder_filters[2], decoder_filters[3], att_heads[3], dpr[3])
        self.Dblock_out_1 = DS_out(decoder_filters[3], 4)
        # model Contour
        self.bottleneck2 = MTBlock_encoder_bottleneck("bottleneck", decoder_filters[0], decoder_filters[0], att_heads[0], dpr[0])
        self.Dblock_2_1 = MTBlock_decoder(decoder_filters[0], decoder_filters[1], att_heads[1], dpr[1])
        self.Dblock_2_2 = MTBlock_decoder(decoder_filters[1], decoder_filters[2], att_heads[2], dpr[2])
        self.Dblock_2_3 = MTBlock_decoder(decoder_filters[2], decoder_filters[3], att_heads[3], dpr[3])
        self.Dblock_out_2 = DS_out(decoder_filters[3], 2)

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, encoder_feature):
        kv = x
        decoder_feature0_task1 = self.bottleneck1(x, kv)
        decoder_feature0_task2 = self.bottleneck2(x, kv)

        kv = self.kv_conv1(kv)
        kv = self.upsample(kv)
        decoder_feature1_task1 = self.Dblock_1_1(decoder_feature0_task1, kv, encoder_feature[0])
        decoder_feature1_task2 = self.Dblock_2_1(decoder_feature0_task2, kv, encoder_feature[0])

        kv = self.kv_conv2(kv)
        kv = self.upsample(kv)
        decoder_feature2_task1 = self.Dblock_1_2(decoder_feature1_task1, kv, encoder_feature[1])
        decoder_feature2_task2 = self.Dblock_2_2(decoder_feature1_task2, kv, encoder_feature[1])

        kv = self.kv_conv3(kv)
        kv = self.upsample(kv)
        decoder_feature3_task1 = self.Dblock_1_3(decoder_feature2_task1, kv, encoder_feature[2])
        decoder_feature3_task2 = self.Dblock_2_3(decoder_feature2_task2, kv, encoder_feature[2])

        task1_feature = [decoder_feature0_task1, decoder_feature1_task1, decoder_feature2_task1, decoder_feature3_task1]
        task2_feature = [decoder_feature0_task2, decoder_feature1_task2, decoder_feature2_task2, decoder_feature3_task2]

        # decoder_feature = [decoder_feature0, decoder_feature1, decoder_feature2, decoder_feature3]
        # return decoder_feature
        # x_out = self.Dblock_out(decoder_feature3)
        return task1_feature, task2_feature

