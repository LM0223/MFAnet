# from .modules.basic import Conv2dBn,Conv2dBnRelu
# import torchvision
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
import lib.resnet_for_PAN_4  as models
'''
Global Attention Upsample Module
'''



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()


        out = identity * a_w * a_h

        return out



#FPA上部分+ARM+FFM+BGALayer

class Conv2dBn(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


'''
	与torchvision.models.resnet中的BasicBlock不同，
	其没有dilation参数,无法组成Dilated ResNet
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3,
                                  stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.conv2 = Conv2dBn(in_ch, out_ch, kernel_size=3,
                              stride=1, padding=dilation, dilation=dilation, bias=False)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv2 = Conv2dBnRelu(out_ch, out_ch, kernel_size=3, stride=stride,
                                  padding=dilation, dilation=dilation, bias=False)

        self.conv3 = Conv2dBn(out_ch, out_ch * 4, kernel_size=1, bias=False)

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class AttentionRefineModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionRefineModule, self).__init__()
        self.conv3x3 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(out_ch, out_ch, kernel_size=1, stride=1,  padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.conv3x3(x)
        attention = self.channel_attention(x)
        # print(x )
        x = x * attention
        y_up=nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)
        # print(x)
        x=y_up
        return x



class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=2, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.conv1x1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride =1, padding =0)      #in->out,尺寸不变
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        #把特征图变成1个数
            Conv2dBnRelu(out_ch, out_ch//reduction, kernel_size=1, stride =1, padding =0),  #out  ->  out/2  ,尺寸不变
            Conv2dBnRelu(out_ch//reduction, out_ch, kernel_size=1, stride =1, padding =0),  #out/2 -> out   ，尺寸不变
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)     #通道
        out = self.conv1x1(fusion)      #in_ch = x1通道数 + x2通道数    ，输出通道out_ch
        attention = self.channel_attention(out) #out —> out
        out = out + out * attention

        return out


class ppp(nn.Module ):
    def __init__ (self,in_ch=1024,out_ch=1024):
        super (ppp, self ).__init__()
        self.conv0 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3, stride=1,padding=3, dilation=3)
        self.conv1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=5, stride=1,padding=6, dilation=6)
        self.conv2 = Conv2dBnRelu(in_ch, out_ch, kernel_size=7, stride=1,padding=7, dilation=7)


    def forward(self,x) :
        h, w = x.size(2), x.size(3)
        x1=self.conv0 (x)
        # print(x1.shape)
        x2=self.conv1(x)
        # print(x2.shape)
        x3 = self.conv2(x)
        # print(x3.shape)
        x4=nn.Upsample(size=(h//2, w//2), mode='bilinear', align_corners=True)(x1)
        x5 = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x2)
        x6 = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x3)
        x7=x4+x5+x6

        return x7



class MBA1(nn.Module):

    def __init__(self):
        super(MBA1, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1,
                padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                64, 64, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.p = ppp(64, 64)
        # self.down = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.right1 = nn.Sequential(
            nn.Conv2d(
                512, 64, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                512, 64, kernel_size=3, stride=1,
                padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                64, 64, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.p(x_d )
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        return out


class MBA2(nn.Module):

    def __init__(self):
        super(MBA2, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                256,256, kernel_size=3, stride=1,
                padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.p=ppp(256,256)
        self.down=nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.right1 = nn.Sequential(
            nn.Conv2d(
                1024, 256, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                1024, 256, kernel_size=3, stride=1,
                padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        # left2 = self.left2(x_d)
        x1=self.p(x_d )
        left2=self.down(x1)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        return out










class MBA3(nn.Module):

    def __init__(self):
        super(MBA3, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1,
                padding=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                512, 512, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.p=ppp(512,512)
        self.down=nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.right1 = nn.Sequential(
            nn.Conv2d(
                2048, 512, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                2048, 512, kernel_size=3, stride=1,
                padding=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                512, 512, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        x1 = self.p(x_d)
        left2 = self.down(x1)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        return out





class FFU(nn.Module):
    def __init__(self, in_ch=1024,out_ch = 2):
        super(FFU, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),  # 不改变通道，尺寸变成1x1
        #     Conv2dBn(out_ch, out_ch, kernel_size=1, stride=1, padding=0),  # 不改变尺寸
        #     nn.Sigmoid()
        # )
        self.query_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.key_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.value_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.conv2 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1)  # 尺寸不变
        # self.conv3 = Conv2dBnRelu(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)  # 尺寸不变

    # x: low level feature
    # y: high level feature
    def forward(self, x, y):  # 尺寸   in_chxin_ch    通道out_ch
            h, w = x.size(2), x.size(3)
            m_batchsize, C, width, height = y.size()
            proj_query = self.query_conv(y).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
            proj_key = self.key_conv(y).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
            energy = torch.bmm(proj_query, proj_key)  # transpose check
            attention = self.softmax(energy)  # BX (N) X (N)
            proj_value = self.value_conv(y).view(m_batchsize, -1, width * height)  # B X C X N

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, width, height)

            y1 = self.gamma * out+y
            # print(y1.shape)
            y1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y1)
            y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
            x = self.conv2(x)
            # print(x.shape)
            # y = self.conv1(y)
            # print(y.shape)
            z = torch.mul(x, y1)
            # print(z.shape)

            return y_up + z

'''
Feature Pyramid Attention Module
FPAModule2:
	downsample use convolution with stride = 2
'''


# class FPAModule2(nn.Module):  # in_ch=256   out_ch=num_class
#
#     def __init__(self, in_ch, out_ch):
#         super(FPAModule2, self).__init__()
#
#         # global pooling branch
#         self.branch1 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # chicun1x1
#             Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0),  # 不改变尺寸
#             Conv2dBnRelu(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#
#         )
#
#         # midddle branch
#         self.mid = nn.Sequential(
#             Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)  # 不改变尺寸
#         )
#
#         self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)  # 尺寸4x4
#
#         self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)  # 尺寸2x2
#
#         self.down3 = nn.Sequential(
#             Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
#             Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
#         )
#
#         self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
#         self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)
#
#     def forward(self, x):
#         h, w = x.size(2), x.size(3)
#         b1 = self.branch1(x)
#         b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)
#
#         mid = self.mid(x)
#
#         x1 = self.down1(x)
#         x2 = self.down2(x1)
#         x3 = self.down3(x2)
#         x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)
#
#         x2 = self.conv2(x2)
#         x = x2 + x3
#         x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
#
#         x1 = self.conv1(x1)
#         x = x + x1
#         x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)
#
#         x = torch.mul(x, mid)
#         x = x + b1
#         return x


class BR_strip(nn.Module):
    def __init__(self,in_ch):
        super(BR_strip, self).__init__()
        self.conv1x3 =nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=(1,3),stride=1,padding=(0,1)),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(True)
        )
        self.conv3x1 = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=(3,1),stride=1,padding=(1,0)),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(True)
        )
    def forward(self,x):
        out = self.conv3x1(self.conv1x3(x))
        return out

class DFE(nn.Module):  # in_ch=256   out_ch=num_class

    def __init__(self, in_ch, out_ch,reduction):
        super(DFE, self).__init__()
        self.conv1x3 =nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=(1,3),stride=1,padding=(0,1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
        self.conv3x1 = nn.Sequential(
            nn.Conv2d(out_ch,out_ch,kernel_size=(3,1),stride=1,padding=(1,0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_ch  // reduction)

        self.conv4 = nn.Conv2d(in_ch, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_ch, kernel_size=1, stride=1, padding=0)

        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)  # 尺寸4x4

        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)  # 尺寸2x2

        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)
        self.conv0 = Conv2dBnRelu(2048, 2, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.conv1x3 (x)
        b2=self.conv3x1 (b1)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b2)

        # mid = self.ca(x)
        # mid=self. conv0(mid)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)

        x2 = self.conv2(x2)
        x4 = x2 + x3
        x5 = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x4)

        x1 = self.conv1(x1)
        x6 = x5 + x1
        x7 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x6)

        # identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv4(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out =  a_w * a_h
        x8 = torch.mul(x7, out)
        x9=x8+b2






        return x9



'''
papers:
	Pyramid Attention Networks
'''








class A_3(nn.Module):
    def __init__(self, layers, pretrained=True, n_class=2):
        '''
        :param backbone: Bcakbone network
        '''
        super(A_3, self).__init__()

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)

        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 1/2
        self.conv2_x = resnet.layer1  # 1/4
        self.conv3_x = resnet.layer2  # 1/8
        self.conv4_x = resnet.layer3  # 1/16
        #self.conv5_x = Bottleneck(1024, 512,  stride=2)  # 1/32
        self.conv5_x = resnet.layer4  # 1/32
        bottom_ch=2048

        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool)  # 1/2
        self.conv2_x = resnet.layer1  # 1/4
        self.conv3_x = resnet.layer2  # 1/8
        self.conv4_x = resnet.layer3  # 1/16
        self.conv5_x = resnet.layer4  # 1/32


        self.fpa = DFE(in_ch=bottom_ch, out_ch=n_class,reduction=64)

        self.gau3 = FFU(in_ch=bottom_ch // 2, out_ch=n_class)

        self.gau2 = FFU(in_ch=bottom_ch // 4, out_ch=n_class)

        self.gau1 = FFU(in_ch=bottom_ch // 8, out_ch=n_class)

        self.gau0 = FFU(in_ch=bottom_ch // 32, out_ch=n_class)


        self.ARM3=AttentionRefineModule(1024,out_ch=bottom_ch // 2)

        self.ARM2 = AttentionRefineModule(in_ch=bottom_ch // 4, out_ch=bottom_ch // 4)

        self.ARM1 = AttentionRefineModule(in_ch=bottom_ch // 8, out_ch=bottom_ch // 8)

        #self.FFM=FeatureFusionModule(in_ch=bottom_ch // 8,out_ch=n_class)
        self.FFM1 = FeatureFusionModule(in_ch=66, out_ch=n_class)

        self.BGA3 = MBA3()
        self.BGA2=MBA2 ()
        self.BGA1=MBA1 ()

        self.conv3 = Conv2dBnRelu(in_ch=n_class, out_ch=n_class, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h, w = x.size(2), x.size(3)
        # stage 1-5
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)
        # for border network
        #x4 = torch.cat([x4, x5], dim=1)
        #print(x4.shape)
        x6 = self.fpa(x5)  # 1/32
        # x7 = self.ARM3(x4)
        x7 = self.BGA3(x3,x5)
        # print(x7.shape)
        x8 = self.gau3(x4, x6)  # 1/16
        x9=self.BGA2 (x2,x4)
        # print(x9.shape)
        # x9 = self.ARM2(x3)
        x10 = self.gau2(x7, x8)  # 1/8
        x11=self.gau1 (x9,x10)
        #print(x10.shape)
        #print(x10.shape)
        # x11=self.ARM1(x2)
        #print(x11.shape)
        x12 = self.BGA1(x1,x7)
        # x13 = self.gau1(x12, x10)  # 1/4
        #print(x2.shape)
        x14 = nn.Upsample(size=(h//4, w//4), mode='bilinear', align_corners=True)(x11)
        # x15 = self.FFM1 (x12,x14)
        x15=self.gau0 (x12,x14)
        #print(x1.shape)
        x16=nn.Upsample(size=(h//4, w//4), mode='bilinear', align_corners=True)(x8)
        x17 = nn.Upsample(size=(h//4, w//4), mode='bilinear', align_corners=True)(x10)
        x18 = nn.Upsample(size=(h//4, w//4), mode='bilinear', align_corners=True)(x11)
        x19=x16+x17+x18
        x20=self.conv3(x19)
        x21=x15+x20
        out = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x21)
        return out
        # out = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x15)
        # return out




if __name__ == '__main__':
    import sys, os
    path = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(path)

    from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
    model = A_3(layers=50, pretrained=False, n_class=2)
    batch = torch.FloatTensor(1, 3, 256, 256)

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(batch)

    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))