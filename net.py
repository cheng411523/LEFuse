import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)



class Block(nn.Module):
    def __init__(self, dim_in, dim_out,mlp_ratio=3):
        super().__init__()
        self.dwconv = ConvBN(dim_in, dim_in, 7, 1, (7 - 1) // 2, groups=dim_in, with_bn=True)
        self.f1 = ConvBN(dim_in, mlp_ratio * dim_in, 1, with_bn=False)
        self.f2 = ConvBN(dim_in, mlp_ratio * dim_in, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim_in, dim_out, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim_out, dim_out, 7, 1, (7 - 1) // 2, groups=dim_out, with_bn=False)
        self.act = nn.ReLU6()
        self.aux_g = ConvBN(dim_in, dim_out, 1, with_bn=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x) 
        x1, x2 = self.f1(x), self.f2(x)  
        x = self.act(x1) * x2  
        x = self.dwconv2(self.g(x)) 
        input = self.aux_g(input)
        x = input + x
        return x
    

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x




class DualBranch_module(nn.Module):
    def __init__(self,dim):
        super().__init__()

        self.num_heads = int(dim / 8)
        self.Global = TransformerBlock(dim=dim, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False,
                                    LayerNorm_type='WithBias')
        self.Local = Block(dim,dim)

        self.se_conv1 = nn.Conv2d(int(dim * 2),int(dim * 2),kernel_size=3,stride=1,padding=1)
        self.se_conv2 = nn.Conv2d(int(dim * 2),int(dim * 2),kernel_size=1,stride=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.se_conv3 = nn.Conv2d(int(dim * 2),dim,kernel_size=1,stride=1)
        self.se_conv4 = nn.Conv2d(int(dim * 2),dim,kernel_size=1,stride=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        res = x
        x_g = self.Global(x)
        x_l = self.Local(x)

        x = torch.cat([x_g,x_l],dim=1)
        x = self.se_conv1(x)
        x = self.se_conv2(x)
        x = self.avg(x)
        return res + x_l * self.sigmoid(self.se_conv4(x)) + x_g * self.sigmoid(self.se_conv3(x))




class up_block(nn.Module):
    def __init__(self,dim_in,dim_out) -> None:
        super().__init__()
        self.up =  nn.Sequential(
            nn.ConvTranspose2d(dim_in,dim_out, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        
    def forward(self,x,tgt):
        x = self.up(x)
        if x.shape != tgt.shape:
            x = F.interpolate(x, size=(tgt.shape[2], tgt.shape[3]), mode='bilinear')
        return x



class Unet_fuser(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        
        self.emb1 = nn.Conv2d(2,dim,kernel_size=3,stride=1,padding=1)
        self.blk1 = DualBranch_module(dim)
        
        self.emb2 = nn.Conv2d(dim,int(dim * 2),kernel_size=3,stride=2,padding=1)
        self.blk2 = DualBranch_module(int(dim * 2))       
        
        self.emb3 = nn.Conv2d(int(dim * 2),int(dim * 3),kernel_size=3,stride=2,padding=1)
        self.blk3 = DualBranch_module(int(dim * 3))    
        
        self.emb4 = nn.Conv2d(int(dim * 3),int(dim * 4),kernel_size=3,stride=2,padding=1)
        self.blk4 = DualBranch_module(int(dim * 4))    
        
        
        
        self.up4 = up_block(int(dim * 4),int(dim * 3))
        self.conv_3_1 = nn.Conv2d(int(dim * 6),int(dim * 3),kernel_size=3,stride=1,padding=1)
        self.upblk1 = DualBranch_module(int(dim * 3))
        

        self.up3 = up_block(int(dim * 3),int(dim * 2))
        self.conv_3_2 = nn.Conv2d(int(dim * 4),int(dim * 2),kernel_size=3,stride=1,padding=1)
        self.upblk2 = DualBranch_module(int(dim * 2))
             
        self.up2 = up_block(int(dim * 2),dim)
        self.conv_3_3 = nn.Conv2d(int(dim * 2),dim,kernel_size=3,stride=1,padding=1)
        self.upblk3 = DualBranch_module(dim)
        
        self.upblk4 = nn.Sequential(
            DualBranch_module(dim),
            DualBranch_module(dim)
        )
        
        
        self.dec = nn.Sequential(
            nn.Conv2d(dim,int(dim * 0.5),kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(dim * 0.5),1,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid()
        )
    
    
    
    def forward(self,vi,ir):
        x = torch.cat([vi,ir],dim=1)
        
        x1 = self.blk1(self.emb1(x)) 
        x2 = self.blk2(self.emb2(x1))
        x3 = self.blk3(self.emb3(x2))
        x4 = self.blk4(self.emb4(x3))
        
        
        out = self.upblk1(self.conv_3_1(torch.cat([self.up4(x4,x3),x3],dim=1))) 
        out = self.upblk2(self.conv_3_2(torch.cat([self.up3(out,x2),x2],dim=1))) 
        out = self.upblk3(self.conv_3_3(torch.cat([self.up2(out,x1),x1],dim=1)))      
        out = self.dec(self.upblk4(out))
        return out
    
    






"""
    Build the model
"""
from vgg import PerceptualLossVgg16ExDark

class LEFuse(nn.Module):
    def __init__(self,):
        super(LEFuse, self).__init__()
        self.fuser = Unet_fuser(dim=32)

        # self.grad = Grad_loss()
        self.content = PerceptualLossVgg16ExDark() # loss

    def forward(self,vi,ir):
        return self.fuser(vi,ir)