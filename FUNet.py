import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import cal_Fda as fda
from utils import *

parser = get_parser()
args = parser.parse_args()

class FUNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4):
        super(FUNet, self).__init__()
        self.conv_down1 = nn.ModuleList([self.convblock(input_ch, W, 1)] + [self.convblock(W * (3**i), W * (3**i), 1) for i in range(1, D)])
        self.conv_down3 = nn.ModuleList([self.convblock(input_ch, W, 3)] + [self.convblock(W * (3**i), W * (3**i), 3) for i in range(1, D)])
        self.conv_down5 = nn.ModuleList([self.convblock(input_ch, W, 5)] + [self.convblock(W * (3**i), W * (3**i), 5) for i in range(1, D)])
        self.conv_weight = nn.ModuleList([
                nn.Conv2d(input_ch, 1, kernel_size=1)] + [nn.Conv2d(W * (3**i), 1, kernel_size=1) for i in range(1, D)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(W * (3**D), W * (3**D), kernel_size=1, stride=1, padding=0,padding_mode='replicate'),
            nn.ReLU(),
        )
        self.bottleneck_weight = nn.Sequential(
            nn.Conv2d(W * (2**D), 1, kernel_size=1),
            nn.ReLU(),
        )
        self.conv_up1 = nn.ModuleList([self.upconvblock(W * (3**i), W * (3**i), 1) for i in range(D, 0, -1)])
        self.conv_up3 = nn.ModuleList([self.upconvblock(W * (3**i), W * (3**i), 3) for i in range(D, 0, -1)])
        self.conv_up5 = nn.ModuleList([self.upconvblock(W * (3**i), W * (3**i), 5) for i in range(D, 0, -1)])
        self.conv_joint = nn.ModuleList([self.convblock(W * (3**i)*2, W * (3**i // 3), 1) for i in range(D, 0, -1)])

        self.conv_out = nn.Sequential(
            nn.Conv2d(W, W, kernel_size=1, stride=1, padding=0,padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(W, output_ch, kernel_size=1, stride=1, padding=0,padding_mode='replicate'),
            nn.ReLU()
        )
        
    def convblock(self, in_ch,out_ch,ksz):
        psz = (ksz-1)//2
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ksz, stride=1, padding=psz,padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=ksz, stride=1, padding=psz,padding_mode='replicate'),            
            nn.ReLU(),
        )
        return block
    
    def upconvblock(self,in_ch,out_ch,ksz):
        psz = (ksz-1)//2
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=ksz, stride=1, padding=psz,padding_mode='replicate'),
            nn.ReLU(),
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
    
    def forward(self, x, focal_length, f_number=None, pixel_size=None):
        Na_yellow = 0.589
        col = torch.tensor([0.66,0.55,0.44]).cuda()
        B, FS, C, H, W = x.shape
        focus_dist = x[:,:,-1].unsqueeze(2).clone() # BxFS C H W
        h = x[:,:,:-1].clone() # BxFS C H W
        focus_dist = focus_dist.expand_as(h)
        r = fda.cal_r(l=Na_yellow,f=focal_length)
        di = fda.cal_di(f=focal_length,F=focus_dist)
        f = fda.cal_f(l=col,r=r).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(di)
        # print(f.shape)
        # print(di.shape)
        Fd = fda.cal_F(f=f,di=di)
        h, Fd = h.reshape(B*FS*(C-1),1,H,W), Fd.reshape(B*FS*(C-1),1,H,W)
        
        grad_x = self.gradient_x(h)
        grad_y = self.gradient_y(h)
        diver = self.diver(h)
        # unloader(grad[0].to('cpu')).save("sample_grad.png")
        # exit()
        grad_x, grad_y, diver = F.normalize(grad_x,p=2,dim=0, eps=1e-12), F.normalize(grad_y,p=2,dim=0, eps=1e-12), F.normalize(diver,p=2,dim=0, eps=1e-12)
        h = torch.cat((h,grad_x,grad_y,diver,Fd),1)
        
        h_s = []
        for i, l in enumerate(self.conv_down1):
            # print(h.shape)
            h1 = self.conv_down1[i](h)
            h2 = self.conv_down3[i](h)
            h3 = self.conv_down5[i](h) # BxFS C H/2**i W/2**i
            h = torch.cat([h1,h2,h3], dim=1)
            # w_h = h.view(B*FS, (C-1), *h.shape[-3:]) # B FS C H/2**i W/2**i
            h_s.append(h.clone()) # B C H/2**i W/2**i
            h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # BxFS C H/2**(i+1) W/2**(i+1)
            # pool_h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # BxFS C H/2**(i+1) W/2**(i+1)
            # h_s.append(torch.max(w_h, dim=1)[0]) # B C H/2**i W/2**i
            # Global Operation
            # stack_pool = pool_h.view(B*FS, (C-1), *pool_h.shape[-3:])
            # pool_max = torch.max(stack_pool, dim=1)[0].unsqueeze(1).expand_as(stack_pool).contiguous().view(B*FS*(C-1), *pool_h.shape[-3:])
            # h = torch.cat([pool_h, pool_max], dim=1)

        h = self.bottleneck(h)
        # w_h = h.view(B*FS, (C-1), *h.shape[-3:]) # B FS C H W
        # h = torch.max(w_h, dim=1)[0]
        
        for i, l in enumerate(self.conv_up1):
            h = self.conv_up1[i](h)
            skip_h = h_s.pop(-1)
            h = self.conv_joint[i](torch.cat([h, skip_h], dim=1))
        
        # output = self.conv_out(h).view(B,FS*(C-1), *h.shape[-2:])
        
        sigma = self.conv_out(h).view(B*FS, C-1, *h.shape[-2:])
        
        d_tuple = fda.cal_d(sigma=sigma,f=focal_length,F=focus_dist.view(B*FS,C-1,H,W),n=f_number,p=pixel_size)
        
        rgb_d = fda.cmp(d_tuple[:,0,:,:],d_tuple[:,1,:,:],d_tuple[:,2,:,:])
        
        recon_sigma = fda.eval_sigma(sigma,rgb_d,col,focal_length,focus_dist.view(B*FS,C-1,H,W),f_number,pixel_size)
        
        
        criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                    ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
        
        loss_l1 = 0
        for i in range(C-1):
            loss_l1 += criterion['l1'](recon_sigma[:,:,i],sigma[:,i].unsqueeze(1).expand(B*FS,C-1,*recon_sigma.shape[-2:]))
        loss_l1 /= C-1
        
        output = rgb_d.view(B,FS*(C-1),H,W)
        
        return output,loss_l1
        # return output
