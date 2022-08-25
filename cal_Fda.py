import math
import numpy as np
import torch
A1=1.03961212
A2=0.231792344
A3=1.01046945
B1=6.00069867e-3
B2=2.00179144e-2
B3=1.03560653e2

def cal_n(l):
    # print(l)
    # print(A1*l**2/(l**2-B1)+A2*l**2/(l**2-B2)+A3*l**2/(l**2-B3)+1)
    # exit()
    n = torch.sqrt(A1*l**2/(l**2-B1)+A2*l**2/(l**2-B2)+A3*l**2/(l**2-B3)+torch.tensor(1))
    return n

def cal_r(l,f):
    n = cal_n(l)
    r = 1/(n-1)/f
    return r

def cal_F(f,di):
    # F = (-2*A**2*di*(N**2+1)/(A**2+di**2)+math.sqrt(A**4*(N**2+1)/(A**2+di**2)+A**2*di**2*(N**2+1)/(A**2+di**2)-A**2))/(2*A**2*(N**2+1)/(A**2+di**2)-2)
    deF = 1/f - 1/di
    F = 1/deF
    return F

def cal_di(f,F):
    # F = (-2*A**2*di*(N**2+1)/(A**2+di**2)+math.sqrt(A**4*(N**2+1)/(A**2+di**2)+A**2*di**2*(N**2+1)/(A**2+di**2)-A**2))/(2*A**2*(N**2+1)/(A**2+di**2)-2)
    de_di = 1/f-1/F
    di = 1/de_di
    return di

def cal_f(l,r):
    n = cal_n(l)
    de_f = (n-1)*r
    f = 1/de_f
    return f

def cal_d(sigma,f,F,n=None,p=None,l=None):
    
    r = cal_r(l=0.589,f=f)
    di = cal_di(f=f,F=F)
    
    if l is None:
        l = [0.66,0.55,0.44]
    l = torch.tensor(l).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    l = l.expand_as(sigma)
    if not n:
        n = [2.8]
    n=torch.tensor(n).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    n = n.expand_as(sigma)
    
    r = torch.tensor(r).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    r = r.expand_as(sigma)

    # di = torch.tensor(di).flatten().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    di = di.expand_as(sigma)
    if not p:
        p = [9e-6]
    p = torch.tensor(p).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    p = p.expand_as(sigma)

    N2 = 1 + A1*l**2/(l**2-B1)+A2*l**2/(l**2-B2)+A3*l**2/(l**2-B3)
    N = torch.sqrt(N2)
    de_f = (N-1)*r
    f = 1/de_f
    deF = de_f - 1/di
    F = 1/deF
    A = f / n
    d1 = A*F*f/(A*f-2*p*sigma*(F-f)+1e-8)
    d2 = A*F*f/(A*f+2*p*sigma*(F-f)+1e-8)
    d = torch.cat([d1.unsqueeze(-1),d2.unsqueeze(-1)],-1)
    return d

def cmp(dr,dg,db):

    d_rgb = torch.cat([dr.unsqueeze(-4),dg.unsqueeze(-4),db.unsqueeze(-4)],-4)

    rg = torch.abs(dr-dg)
    rb = torch.abs(dr-db)
    bg = torch.abs(db-dg)
    
    s_a = (rg+rb+bg)/2
    s_m = torch.argmin(s_a,-1,keepdim=True).long()
    s_m = s_m.unsqueeze(-4).expand(list(d_rgb.shape[:-1])+[1])

    # print(d_rgb.shape)
    # print(s_m.shape)
    
    d = torch.gather(d_rgb,-1,s_m).squeeze()
    # d = (d-torch.min(d))/(torch.max(d)-torch.min(d))
    # print(d)
    return d

def cal_sigma(d,f,F,n=None,p=None,l=None):
    
    r = cal_r(l=0.589,f=f)
    di = cal_di(f=f,F=F)
    
    if l is None:
        l = [0.66,0.55,0.44]
    l = torch.tensor(l).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    l = l.expand_as(d)
    if not n:
        n = [2.8]
    n=torch.tensor(n).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    n = n.expand_as(d)
    
    r = torch.tensor(r).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    r = r.expand_as(d)
    
    di = torch.tensor(di).cuda()
    di = di.expand_as(d)
    if not p:
        p = [9e-6]
    p = torch.tensor(p).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    p = p.expand_as(d)

    N2 = 1 + A1*l**2/(l**2-B1)+A2*l**2/(l**2-B2)+A3*l**2/(l**2-B3)
    N = torch.sqrt(N2)
    de_f = (N-1)*r
    f = 1/de_f
    deF = de_f - 1/di
    F = 1/deF
    A = f / n
    sigma = A/(2*p)*torch.abs(d-F)/d*f/(F-f)
    sigma = A*torch.abs(d-F)*f/(2*p*d*(F-f)+1e-8)
    return sigma

def eval_sigma(sigma,do,l,f,F,n,p):
    
    B,C,H,W = do.shape
    r = cal_r(l=0.589,f=f)
    di = cal_di(f=f,F=F)
    # N = cal_n(l).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(do)
    f = cal_f(l,r).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(do)
    F = cal_F(f,di)
    A = f / n
    
    # do = A*di/(A*(di*r*(N-1)-1)-2*p*sigma)
    
    A,f,F,do = A.unsqueeze(-3).expand(B,C,C,H,W),f.unsqueeze(-3).expand(B,C,C,H,W),F.unsqueeze(-3).expand(B,C,C,H,W),do.unsqueeze(-3).expand(B,C,C,H,W)
    
    recon_sigma = A*f/(2*p*(F-f))*torch.abs(1-F/do)
    for i in range(C):
        recon_sigma[:,i,i] = sigma[:,i].clone()
    
    return recon_sigma

# print(cal_n(0.5893))
# print(cal_r(0.5893,2.4e4))
# print(cal_F(8.063564002660449e-05,1.52627,2.535e4))