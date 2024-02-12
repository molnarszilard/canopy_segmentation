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
    def __init__(self, pretrained=True, fixed_feature_weights=False,size='medium'):
        super(NetworkModule, self).__init__()
        self.size = size
        resnet = resnet101(pretrained=pretrained)

        # Freeze those weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        if self.size in ['medium','large']:
            self.layer3 = nn.Sequential(resnet.layer3)
        if self.size in ['large']:
            self.layer4 = nn.Sequential(resnet.layer4)

        # lateral layers
        if self.size in ['large']:
            self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        if self.size in ['medium','large']:
            self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        if self.size in ['large']:
            self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        if self.size in ['medium','large']:
            self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        if self.size in ['large']:
            self.agg1 = agg_node(256, 64)
        if self.size in ['medium','large']:
            self.agg2 = agg_node(256, 64)
        self.agg3 = agg_node(256, 64)
        self.agg4 = agg_node(256, 64)
        self.agg5 = agg_node(256, 64)
        
        # Upshuffle layers
        if self.size in ['large']:
            self.up1 = upshuffle(64,64,16)
        if self.size in ['medium','large']:
            self.up2 = upshuffle(64,64,8)
        self.up3 = upshuffle(64,64,4)
        self.up4 = upshuffle(64,64,2)
        self.up5 = upshuffle(64,64,2)
        
        if self.size in ['large']:
            self.predict1 = smooth(320, 64)
        if self.size in ['medium']:
            self.predict1 = smooth(256, 64)
        if self.size in ['small']:
            self.predict1 = smooth(192, 64)
        self.predict2 = predict(64, 1)
        
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

    def get_size(self):
        return self.size

    def forward(self, x):
        _,_,H,W = x.size()
        
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1) 
        c3 = self.layer2(c2)
        if self.size in ['medium','large']:
            c4 = self.layer3(c3)
        if self.size in ['large']:
            c5 = self.layer4(c4)
            

        # Top-down
        if self.size in ['large']:
            # print(c5.shape)
            p5 = self.toplayer(c5)
            # print(p5.shape)
            p5b = self.latlayer1(c4)
            # print(p5b.shape)
            p4 = self._upsample_add(p5, p5b)
            # print(p4.shape)
            p4 = self.smooth1(p4)
            # print(p4.shape)
        if self.size in ['medium','large']:
            if self.size in ['medium']:
                p4 = self.latlayer1(c4)
            p4b = self.latlayer2(c3)
            p3 = self._upsample_add(p4, p4b)
            p3 = self.smooth2(p3)
        if self.size in ['small']:
            p3 = self.latlayer2(c3)
        p3b = self.latlayer3(c2)
        p2 = self._upsample_add(p3, p3b)
        p2 = self.smooth3(p2)
        p2b = self.latlayer4(c1)
        p1 = self._upsample_add(p2, p2b)
        p1 = self.smooth4(p1)
        
        # Top-down predict and refine
        if self.size in ['large']:
            d5b = self.agg1(p5)
            d5 = self.up1(d5b)
        if self.size in ['medium','large']:
            d4b = self.agg2(p4)
            d4 = self.up2(d4b)
        d3b = self.agg3(p3)
        d3 = self.up3(d3b)
        d2b = self.agg4(p2)
        d2 = self.up4(d2b)
        d1b = self.agg5(p1)
        d1 = self.up5(d1b)

        if self.size in ['large']:
            vol = torch.cat( [ F.interpolate(d, size=(H,W), align_corners=False, mode='bilinear') for d in [d5,d4,d3,d2,d1] ], dim=1 )
        if self.size in ['medium']:
            vol = torch.cat( [ F.interpolate(d, size=(H,W), align_corners=False, mode='bilinear') for d in [d4,d3,d2,d1] ], dim=1 )
        if self.size in ['small']:
            vol = torch.cat( [ F.interpolate(d, size=(H,W), align_corners=False, mode='bilinear') for d in [d3,d2,d1] ], dim=1 )
        
        pred1 = self.predict1(vol)
        pred2 = self.predict2(pred1)
        return pred2
