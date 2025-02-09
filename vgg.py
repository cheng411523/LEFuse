import torch
import torchvision
import torch.nn as nn


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False
        
        
class Vgg16ExDark(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16ExDark, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg_pretrained_features = vgg16.features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22] 
        out = []
        for i in range(indices[-1]+1):
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out


class PerceptualLossVgg16ExDark(nn.Module):
    def __init__(self,
                 vgg=None,
                 load_model=None,
                 indices=None,
                 normalize=True):
        super(PerceptualLossVgg16ExDark, self).__init__()
        if vgg is None:
            self.vgg = Vgg16ExDark(load_model)
        else:
            self.vgg = vgg
        self.vgg = self.vgg.cuda()
        self.criter  = nn.L1Loss()
        self.weights = [1.0,2.0,3.0,4.0] #,
        self.indices = indices or [3,8,15,22] #
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self,x,y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criter(x_vgg[i], y_vgg[i].detach())
        return loss
