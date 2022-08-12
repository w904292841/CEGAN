import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F

class FUNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4):
        super(FUNet, self).__init__()
        self.conv_down = nn.ModuleList([self.convblock(input_ch, W)] + [self.convblock(W * (2**i), W * (2**i)) for i in range(1, D)])
        self.conv_weight = nn.ModuleList([
                nn.Conv2d(input_ch, 1, kernel_size=1)] + [nn.Conv2d(W * (2**i), 1, kernel_size=1) for i in range(1, D)])
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
            nn.Conv2d(W, W, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(W, output_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        return block
    
    def laplacian_kernel(self):
        kernel = [[0,1,0],[1,-4,1],[0,1,0]]
        kernel = np.array(kernel)
        return torch.from_numpy(kernel).float().view(1,1,3,3)

    def gradient(self, inp):
        D_dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx + D_dy

    def forward(self, x, focus_dist):
        B, FS, C, H, W = x.shape
        h = x.view(B*FS, C, H, W) # BxFS C H W
        focus_dist = focus_dist.view(B*FS,1,1,1).expand_as(h[:,0,:,:].unsqueeze(1))
        h = torch.cat((h,focus_dist),1)
        h_s = []
        for i, l in enumerate(self.conv_down):
            h = self.conv_down[i](h) # BxFS C H/2**i W/2**i
            # print(h)
            # exit()
            pool_h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # BxFS C H/2**(i+1) W/2**(i+1)
            w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H/2**i W/2**i
            h_s.append(torch.max(w_h, dim=1)[0]) # B C H/2**i W/2**i
            # Global Operation
            stack_pool = pool_h.view(B, FS, *pool_h.shape[-3:])
            pool_max = torch.max(stack_pool, dim=1)[0].unsqueeze(1).expand_as(stack_pool).contiguous().view(B*FS, *pool_h.shape[-3:])
            h = torch.cat([pool_h, pool_max], dim=1)

        h = self.bottleneck(h)
        w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H W
        h = torch.max(w_h, dim=1)[0]
        
        for i, l in enumerate(self.conv_up):
            h = self.conv_up[i](h)
            skip_h = h_s.pop(-1)
            h = self.conv_joint[i](torch.cat([h, skip_h], dim=1))
        
        output = self.conv_out(h)
        # print(output.shape)
        # exit()
        return output[:,-1,:,:]
