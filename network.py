import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.resnet import resnet101

def agg_node(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid(),
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )

class NetworkModule(nn.Module):
    def __init__(self, pretrained=True, fixed_feature_weights=False):
        super(NetworkModule, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        # Freeze those weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        # self.layer3 = nn.Sequential(resnet.layer3)
        # self.layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        # self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        # self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer5 = nn.Conv2d( 3, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        # self.agg1 = agg_node(256, 64)
        # self.agg2 = agg_node(256, 64)
        self.agg3 = agg_node(256, 64)
        self.agg4 = agg_node(256, 64)
        self.agg5 = agg_node(256, 64)
        # self.agg6 = agg_node(256, 64)
        
        # Upshuffle layers
        # self.up1 = upshuffle(64,64,16)
        # self.up2 = upshuffle(64,64,8)
        self.up3 = upshuffle(64,64,4)
        self.up4 = upshuffle(64,64,2)
        self.up5 = upshuffle(64,64,2)
        # self.up6 = upshuffle(64,64,1)
        
        self.predict1 = smooth(192, 64)
        self.predict2 = predict(64, 1)

        self.activation = nn.Sigmoid()
        # self.activation = nn.Threshold(threshold=0.5, value=0.0)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), align_corners=False, mode='bilinear') + y

    def forward(self, x):
        _,_,H,W = x.size()
        
        # Bottom-up

        c1 = self.layer0(x)
        c2 = self.layer1(c1) 
        c3 = self.layer2(c2)
        # c4 = self.layer3(c3)
        # c5 = self.layer4(c4)

        # Top-down
        # p5 = self.toplayer(c5)
        # p5b = self.latlayer1(c4)
        # p4 = self._upsample_add(p5, p5b)
        # p4 = self.smooth1(p4)
        # p4 = self.latlayer1(c4)
        # p4b = self.latlayer2(c3)
        # p3 = self._upsample_add(p4, p4b)
        # p3 = self.smooth2(p3)
        p3 = self.latlayer2(c3)
        p3b = self.latlayer3(c2)
        p2 = self._upsample_add(p3, p3b)
        p2 = self.smooth3(p2)
        p2b = self.latlayer4(c1)
        p1 = self._upsample_add(p2, p2b)
        p1 = self.smooth4(p1)
        # p1b = self.latlayer5(x)
        # p0 = self._upsample_add(p1, p1b)
        # p0 = self.smooth5(p0)
        
        # Top-down predict and refine
        # d5b = self.agg1(p5)
        # d5 = self.up1(d5b)
        # d4b = self.agg2(p4)
        # d4 = self.up2(d4b)
        d3b = self.agg3(p3)
        d3 = self.up3(d3b)
        d2b = self.agg4(p2)
        d2 = self.up4(d2b)
        d1b = self.agg5(p1)
        d1 = self.up5(d1b)
        # d0 = self.agg6(p0)
        
        # _,_,H,W = d2.size()
        vol = torch.cat( [ F.interpolate(d, size=(H,W), align_corners=False, mode='bilinear') for d in [d3,d2,d1] ], dim=1 )

        # d5, d4, d3, d2 = self.up1(self.agg1(p5)), self.up2(self.agg2(p4)), self.up3(self.agg3(p3)), self.agg4(p2)
        # _,_,H,W = d2.size()
        # vol = torch.cat( [ F.upsample(d, size=(H,W), mode='bilinear') for d in [d5,d4,d3,d2,d1] ], dim=1 )
        
        pred1 = self.predict1(vol)
        # pred2 = F.interpolate(self.predict2(pred1), size=(H*2,W*2), align_corners=False, mode='bilinear')
        # def hard_swish(x):
        # return x * tf.nn.relu6(x + 3) / 6
        pred2 = self.predict2(pred1)
        pred3 = self.activation(pred2)
        return pred3
