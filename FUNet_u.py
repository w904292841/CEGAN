import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from utils import *
import cal_Fda as fda
import numpy as np
from PIL import Image
import torchvision
# from loss import BlurMetric

parser = get_parser()
args = parser.parse_args()
unloader = torchvision.transforms.ToPILImage()

class FUNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4):
        super(FUNet, self).__init__()
        self.conv_down = nn.ModuleList([self.convblock(input_ch, W)] + [self.convblock(W * (2**i), W * (2**(i+1))) for i in range(0, D)])
        self.conv_weight = nn.ModuleList([
                nn.Conv2d(input_ch, 1, kernel_size=1)] + [nn.Conv2d(W * (2**i), 1, kernel_size=1) for i in range(0, D)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(W * (2**D), W * (2**D), kernel_size=1, stride=1, padding=0,padding_mode='replicate'),
            nn.Softplus(),
        )
        self.bottleneck_weight = nn.Sequential(
            nn.Conv2d(W * (2**D), 1, kernel_size=1),
            nn.Softplus(),
        )
        self.conv_up = nn.ModuleList([self.upconvblock(W * (2**i), W * (2**i) // 2) for i in range(D, 0, -1)])
        self.conv_joint = nn.ModuleList([self.convblock(W * (2**i), W * (2**i // 2)) for i in range(D, 0, -1)])

        self.conv_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(W, W, kernel_size=1, stride=1, padding=0,padding_mode='replicate'),
            nn.Softplus(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(W, output_ch, kernel_size=1, stride=1, padding=0,padding_mode='replicate'),
            nn.Softplus()
        )
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def convblock(self, in_ch,out_ch):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0,padding_mode='replicate'),
            nn.Softplus(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0,padding_mode='replicate'),            
            nn.Softplus(),
        )
        return block
    
    def upconvblock(self,in_ch,out_ch):
        block = nn.Sequential(
            
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,padding_mode='replicate'),
            nn.Softplus()
        )
        return block
    
    def gradient_x(self, inp):
        D_dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx

    def gradient_y(self, inp):
        D_dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
        return D_dy

    def diver(self,inp):
        diver_x = self.gradient_x(self.gradient_x(inp))
        diver_y = self.gradient_y(self.gradient_y(inp))
        return diver_x**2+diver_y**2

    def forward(self, x, focus_length = None, focal_distance = None, f_number = None, pixel_size = None):
        wav_len = [0.66,0.55,0.44]
        Na_yellow = 0.589
        B, FS, C, H, W = x.shape
        h = x.view(B*FS*C, 1, H, W) 
        # unloader(diver[0].to('cpu')).save("sample_diver.png")
        grad_x = self.gradient_x(h)
        grad_y = self.gradient_y(h)
        diver = self.diver(h)
        # unloader(grad[0].to('cpu')).save("sample_grad.png")
        # exit()
        h = torch.cat((h,grad_x,grad_y,diver),1)
        h_s = []
        pool_h = [h]
        skip_h = h
        for i, l in enumerate(self.conv_down):
            h = self.conv_down[i](h) # B C H/2**i W/2**i
            h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # B C H/2**(i+1) W/2**(i+1)
            # skip_h = F.max_pool2d(skip_h, kernel_size=2, stride=2, padding=0)
            # skip_h = torch.cat([skip_h,skip_h],1)
            # # pool_h = F.max_pool2d(pool_h, kernel_size=2, stride=2, padding=0)
            # # pool_h = torch.cat((pool_h,pool_h),1)
            # h = h+skip_h
            pool_h += [h.clone()]
            
            # w_h = h.view(B*C, *h.shape[-3:]) # B C H/2**i W/2**i
            # h_s.append(torch.max(w_h, dim=1)[0]) # B C H/2**i W/2**i
            # Global Operation
            # stack_pool = pool_h.view(B*C, *pool_h.shape[-3:])
            # pool_max = stack_pool
            # h = torch.cat([pool_h, pool_max], dim=1)

        h = self.bottleneck(h)
        # w_h = h.view(B*C, *h.shape[-3:]) # B C H W
        # h = torch.max(w_h, dim=1)[0]
        # upsp_h = pool_h[-1]
        for i, l in enumerate(self.conv_up):
            h = h+pool_h.pop()
            # h = h+upsp_h
            h = self.up_sample(h)
            h = self.conv_up[i](h)
            # upsp_h = h.clone()
            # upsp_h = self.up_sample(upsp_h)[:,:h.shape[1],:,:]
            # skip_h = h_s.pop(-1)
            # h = self.conv_joint[i](torch.cat([h, skip_h], dim=1))
        
        net_out = self.conv_out(h)
        # print(net_out.shape)
        dfc_map = net_out.view(B*FS,C,H*2,W*2)

        d_tuple = fda.cal_d(sigma=dfc_map,f=focus_length,F=focal_distance,n=f_number,p=pixel_size)
        
        rgb_d = fda.cmp(d_tuple[:,0,:,:],d_tuple[:,1,:,:],d_tuple[:,2,:,:])
        
        criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                    ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
        
        cfirst = [0,1,2,0]
        loss_ssim = []
        loss_l1 = []
        for col in range(3):
            loss_ssim += [1 - criterion['ssim'](rgb_d[:,cfirst[col]].unsqueeze(1), rgb_d[:,cfirst[col+1]].unsqueeze(1))]
            loss_l1 += [criterion['l1'](rgb_d[:,cfirst[col]].unsqueeze(1), rgb_d[:,cfirst[col+1]].unsqueeze(1))]

        loss_ssim = sum(loss_ssim)/3
        loss_l1 = sum(loss_l1)/3
        
        pre_d = torch.sum(rgb_d,dim=1)/3
        
        pre_d = pre_d.view(B, FS, H*2, W*2)
        
        # diver = diver.view(B*FS,C,H,W)[:,0]
        # diver = diver / torch.max(diver,0)[0] + 1e-8
        
        # return net_out.view(B, FS, C, H, W)
        # return torch.sum(net_out.view(B, FS, C, H, W),2)/C
        return pre_d,loss_ssim,loss_l1
        # return diver.view(B,FS,H,W),loss_ssim,loss_l1
