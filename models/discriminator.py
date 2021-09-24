import torch.nn as nn
import torch


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        return x


class LightFCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(LightFCDiscriminator, self).__init__()

        # context
        self.conv1 = DepthWiseSeparableConvolution(num_classes, ndf)
        self.conv2 = DepthWiseSeparableConvolution(ndf, ndf*2)
        self.conv3 = DepthWiseSeparableConvolution(ndf*2, ndf*4)
        self.conv4 = DepthWiseSeparableConvolution(ndf*4, ndf*8)
        self.classifier = DepthWiseSeparableConvolution(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        return x


class LightLightFCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(LightLightFCDiscriminator, self).__init__()

        # context
        self.conv1 = DepthWiseSeparableConvolution(num_classes, ndf)
        self.conv2 = DepthWiseSeparableConvolution(ndf, ndf*2)
        self.classifier = DepthWiseSeparableConvolution(ndf*2, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        # x = self.up_sample(x)
        return x


class MultiClassLightFCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(MultiClassLightFCDiscriminator, self).__init__()

        # context
        self.conv1 = DepthWiseSeparableConvolution(num_classes, ndf)
        self.conv2 = DepthWiseSeparableConvolution(ndf, ndf * 2)
        self.conv3 = DepthWiseSeparableConvolution(ndf * 2, ndf * 4)
        self.classifier = DepthWiseSeparableConvolution(ndf * 4, num_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        # x = self.up_sample(x)
        return x


class LightFCDiscriminator2(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(LightFCDiscriminator2, self).__init__()

        # context
        self.conv1 = DepthWiseSeparableConvolution(num_classes, ndf)
        self.conv2 = DepthWiseSeparableConvolution(ndf, ndf*2)
        self.conv3 = DepthWiseSeparableConvolution(ndf*2+ndf, ndf*4)
        self.conv4 = DepthWiseSeparableConvolution(ndf*4+ndf*2, ndf*8)
        self.classifier = DepthWiseSeparableConvolution(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.leaky_relu(x1)

        x2 = self.conv2(x1)
        x2 = self.leaky_relu(x2)

        x3 = self.conv3(torch.cat((x1, x2), dim=1))
        x3 = self.leaky_relu(x3)

        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x4 = self.leaky_relu(x4)

        x5 = self.classifier(x4)
        # x = self.up_sample(x)
        return x5


class DepthWiseSeparableConvolution(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DepthWiseSeparableConvolution, self).__init__()
        self.depth_wise = nn.Conv2d(ch_in, ch_in, kernel_size=4, stride=2, padding=1, groups=ch_in)
        self.point_wise = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out

