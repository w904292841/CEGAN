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
    n = math.sqrt(1+A1*l**2/(l**2-B1)+A2*l**2/(l**2-B2)+A3*l**2/(l**2-B3))
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

def cal_d(sigma,r=None,di=None,n=None,l=None,p=None):
    
    if l is None:
        l = [0.66,0.55,0.44]
    l = torch.tensor(l).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    l = l.expand_as(sigma)
    if not n:
        n = [2.8]
    n=torch.tensor(n).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    n = n.expand_as(sigma)
    if not r:
        r = [80.63564002660449]
    r = torch.tensor(r).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    r = r.expand_as(sigma)
    if di is None:
        di = [2.53e-2]
    di = torch.tensor(di).flatten().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
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
    d1 = A*F*f/(A*f-2*p*sigma*(F-f))
    d2 = A*F*f/(A*f+2*p*sigma*(F-f))
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
# print(cal_n(0.5893))
# print(cal_r(0.5893,2.4e4))
# print(cal_F(8.063564002660449e-05,1.52627,2.535e4))