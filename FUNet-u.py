import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import cal_Fda as fda
import numpy as np
from PIL import Image
import torchvision

unloader = torchvision.transforms.ToPILImage()

class FUNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4):
        super(FUNet, self).__init__()
        self.conv_down = nn.ModuleList([self.convblock(input_ch, W)] + [self.convblock(W * (2**i), W * (2**(i+1))) for i in range(0, D)])
        self.conv_weight = nn.ModuleList([
                nn.Conv2d(input_ch, 1, kernel_size=1)] + [nn.Conv2d(W * (2**i), 1, kernel_size=1) for i in range(0, D)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(W * (2**D), W * (2**D), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.bottleneck_weight = nn.Sequential(
            nn.Conv2d(W * (2**D), 1, kernel_size=1),
            nn.ReLU(),
        )
        self.conv_up = nn.ModuleList([self.upconvblock(W * (2**i), W * (2**i) // 2) for i in range(D, 0, -1)])
        self.conv_joint = nn.ModuleList([self.convblock(W * (2**i), W * (2**i // 2)) for i in range(D, 0, -1)])

        self.conv_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(W, W, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(W, output_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def convblock(self, in_ch,out_ch):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),            
            nn.ReLU(),
        )
        return block
    
    def upconvblock(self,in_ch,out_ch):
        block = nn.Sequential(
            
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        return block
    
    def laplacian_kernel(self):
        kernel = [[0,1,0],[1,-4,1],[0,1,0]]
        kernel = np.array(kernel)
        return torch.from_numpy(kernel).float().view(1,1,3,3)

    def gradient(self, inp):
        D_dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx+D_dy

    def forward(self, x, focus_length = None, focal_distance = None, f_number = None):
        wav_len = [0.66,0.55,0.44]
        Na_yellow = 0.589
        B, FS, C, H, W = x.shape
        h = x.view(B*FS*C, 1, H, W) 
        diver = F.conv2d(h,self.laplacian_kernel().to(torch.get_device(h)),padding="same")
        # unloader(diver[0].to('cpu')).save("sample_diver.png")
        grad = self.gradient(h)
        # unloader(grad[0].to('cpu')).save("sample_grad.png")
        # exit()
        h = torch.cat((h,grad,diver),1)
        h_s = []
        pool_h = h.clone()
        for i, l in enumerate(self.conv_down):
            h = self.conv_down[i](h) # B C H/2**i W/2**i
            h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # B C H/2**(i+1) W/2**(i+1)
            pool_h = F.max_pool2d(pool_h, kernel_size=2, stride=2, padding=0)
            pool_h = torch.cat((pool_h,pool_h),1)
            h += pool_h
            # w_h = h.view(B*C, *h.shape[-3:]) # B C H/2**i W/2**i
            # h_s.append(torch.max(w_h, dim=1)[0]) # B C H/2**i W/2**i
            # Global Operation
            # stack_pool = pool_h.view(B*C, *pool_h.shape[-3:])
            # pool_max = stack_pool
            # h = torch.cat([pool_h, pool_max], dim=1)

        h = self.bottleneck(h)
        # w_h = h.view(B*C, *h.shape[-3:]) # B C H W
        # h = torch.max(w_h, dim=1)[0]
        upsp_h = h.clone()
        for i, l in enumerate(self.conv_up):
            h = self.up_sample(h)
            upsp_h = self.up_sample(upsp_h)[:,:h.shape[1],:,:]
            h += upsp_h
            h = self.conv_up[i](h)
            # skip_h = h_s.pop(-1)
            # h = self.conv_joint[i](torch.cat([h, skip_h], dim=1))
        
        net_out = self.conv_out(h)
        # print(net_out.shape)
        dfc_map = net_out.view(B*FS,C,H,W)

        if focus_length and f_number:
            focal_distance = focal_distance
            r = fda.cal_r(l=Na_yellow,f=focus_length)
            di = fda.cal_di(f=focus_length,F=focal_distance)

            d_tuple = fda.cal_d(sigma=dfc_map,r=r,di=di,n=f_number)
        else:
            d_tuple = fda.cal_d(dfc_map)

        rgb_d = fda.cmp(d_tuple[:,0,:,:],d_tuple[:,1,:,:],d_tuple[:,2,:,:])

        pre_d = torch.sum(rgb_d,dim=1)/3
        
        pre_d = pre_d.view(B, FS, H, W)
        
        # return torch.sum(net_out.view(B, FS, C, H, W),2)/C
        return pre_d
