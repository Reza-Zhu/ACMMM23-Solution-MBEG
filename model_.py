import os
import timm
import time
import math
import torch
import torch.nn as nn
from torch.nn import init, functional
from torchvision import models



class ClassBlock(nn.Module):

    def __init__(self, input_dim, class_num, drop_rate, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [
            nn.Linear(input_dim, num_bottleneck),
            nn.GELU(),
            nn.BatchNorm1d(num_bottleneck),
            nn.Dropout(p=drop_rate)
        ]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):

        x = self.add_block(x)
        feature = x
        x = self.classifier(x)
        return x, feature

class EVA(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=True):
        super(EVA, self).__init__()
        self.model_1 = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',pretrained=True,
    num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',pretrained=True,
    num_classes=0)
        self.classifier = ClassBlock(1024, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
            f1 = None
        else:
            
            x1 = self.model_1(x1)
            y1, f1 = self.classifier(x1)

        if x2 is None:
            y2 = None
            f2 = None
        else:

            x2 = self.model_2(x2)
            y2, f2 = self.classifier(x2)
        if self.training:
            return y1, y2, f1, f2
            # output1, output2
        else:
            return f1, f2

        
class MobileViT(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=True):
        super(MobileViT, self).__init__()
        self.model_1 =timm.create_model('mobilevitv2_200.cvnets_in22k_ft_in1k_384',
                                         pretrained=True, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model('mobilevitv2_200.cvnets_in22k_ft_in1k_384',pretrained=True,
    num_classes=0)
        self.classifier = ClassBlock(1024, classes, drop_rate)
        # self.proj = nn.Conv2d(768, 1024, kernel_size=1, stride=1)
        # self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(1024, 2048, kernel_size=1, bias=False),
        #                                          torch.nn.BatchNorm2d(2048),
        #                                          torch.nn.ReLU())
        # self.norm = torch.nn.LayerNorm(768)

        
#     def hbp(self, conv1, conv2):
#         N = conv1.size()[0]
#         proj_1 = self.bilinear_proj(conv1)
#         proj_2 = self.bilinear_proj(conv2)

#         X = proj_1 * proj_2
#         X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
#         X = X.view(N, 2048)
#         X = torch.sqrt(X + 1e-5)
#         X = torch.nn.functional.normalize(X)
#         return X

#     def restore_vit_feature(self, x):
#         x = x[:, 1:, :]
#         x = rearrange(x, "b (h w) y -> b y h w", h=32, w=32)
#         x = self.proj(x)
#         return x 
    
    def fusion_features(self, x, model):

        y = []
        x = model(x)


        # v_f = self.restore_vit_feature(v_f)
        # l_f = self.restore_vit_feature(x)
        # l_f = self.restore_vit_feature(l_f)

        # HBP softmax 3 layer feature X multiply
        # print(v_f.shape, l_f.shape)
        # x = self.hbp(v_f, l_f)
        # print(x.shape)
        # x_cls = torch.mean(x, dim=1)  # GlobalAvgPool2d
        
        # x_cls = self.norm(x_cls)
        # print(x_cls.shape)
        # direct softmax and Triplet

        if self.training:
            y, f = self.classifier(x)
            return y, f
        else:
            f = self.classifier(x)
            return f


       

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
            f1 = None
            output1 = None
        else:
            y1, f1 = self.fusion_features(x1, self.model_1)

        if x2 is None:
            y2 = None
            f2 = None
            output2 = None
        else:
            y2, f2 = self.fusion_features(x2, self.model_2)

        if self.training:
            return y1, y2, f1, f2
            # output1, output2
        else:
            return f1, f2
    
    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
            f1 = None
            output1 = None
        else:
            y1, f1 = self.fusion_features(x1, self.model_1)

        if x2 is None:
            y2 = None
            f2 = None
            output2 = None
        else:
            y2, f2 = self.fusion_features(x2, self.model_2)

        if self.training:
            return y1, y2, f1, f2
            # output1, output2
        else:
            return f1, f2
    
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    # import ssl

    # ssl._create_default_https_context = ssl._create_unverified_context
    # model = ViT_two_view_LPN(100, 0.1).cuda()
    # model = Hybird_ViT(100, 0.1).cuda()
    # model = ViT_two_view_LPN(100, 0.1).cuda()
    model = EVA(100, 0.1, True)
    # print(model)
    # model = EfficientNet_b()
    # print(model.device)
    # print(model.extract_features)
    # Here I left a simple forward function.
    # Test the model, before you train it.
    input = torch.randn(8, 3, 336, 336)
    output1, output2, f1, f2 = model(input, input)
    print(output1.shape)
    print(f1.shape)
    # print(output)

model_dict = {
    "EVA": EVA,
    "MobileViT":MobileViT
}
