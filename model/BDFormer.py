import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mt_maxvit import MT_MaxViT

from torchsummary import summary


class BDFormer(nn.Module):  # multi-task netowrk with MaxViT_small_Encoder and MTSwin_Decoder
    def __init__(self, img_size=256, in_channels=1, num_classes=21843, zero_head=False, window_size=8, has_dropout=False, vis=False):
        super(BDFormer, self).__init__()
        self.has_dropout = has_dropout
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.window_size = window_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.multi_task_MaxViT = MT_MaxViT(img_size=self.img_size,  # 224
                                patch_size=4,                        # 4
                                in_chans=self.in_channels,           # 3
                                num_classes=self.num_classes,        # 9
                                embed_dim=96,                        # 96
                                depths=[2, 2, 2, 2],                 # [2, 2, 6, 2]
                                # depths_decoder=[2, 2, 2, 1],
                                num_heads=[3, 6, 12, 24],            # [3, 6, 12, 24]
                                window_size=self.window_size,        # 7
                                mlp_ratio=4,                         # 4.
                                qkv_bias=True,                       # True
                                qk_scale=None,                       # None
                                drop_rate=0.0,                       # 0.0
                                drop_path_rate=0.2,                  # 0.1
                                ape=False,                           # False
                                patch_norm=True,                     # True
                                use_checkpoint=False,                # "expand_first"  default = False
                                has_dropout=self.has_dropout)

    def forward(self, x):
        if x.size()[1] == 1:
            x = self.conv(x)
        logits = self.multi_task_MaxViT(x)
        return logits



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Multi_MaxViT(img_size=256, in_channels=3, num_classes=1, window_size=8).to(device)

    img = torch.randn([2, 1, 256, 256]).to(device)
    out1, out2 = net(img)
    print(out1.shape)
    print(out2.shape)
    # summary(net, (3, 256, 256))

