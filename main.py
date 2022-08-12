import torch
import torch.nn as nn
import os
import FUNet as fu
import args as a
import datasets as ds
import fs
import torchvision
import numpy as np
import cv2
import imageio
from datetime import datetime
import torch.nn.functional as F
from loss import BlurMetric
from PIL import Image

parser = a.get_parser()
args = parser.parse_args()
unloader = torchvision.transforms.ToPILImage()

DATA_PARAMS = {
        'DATA_PATH': './data/',
        'DATA_SET': 'fs_',
        'DATA_NUM': 6,
        'FLAG_NOISE': False,
        'FLAG_SHUFFLE': False,
        'INP_IMG_NUM': 1,
        'FLAG_IO_DATA': {
            'INP_RGB': True,
            'INP_COC': False,
            'INP_AIF': False,
            'INP_DIST':True,

            'OUT_COC': False,
            'OUT_DEPTH': True,
        },
        'TRAIN_SPLIT': 0.8,
        'DATASET_SHUFFLE': True,
        'WORKERS_NUM': 4,
        'BATCH_SIZE': 16,
        'DATA_RATIO_STRATEGY': 0,
        'FOCUS_DIST': [0.1,0.15,0.3,0.7,1.5],
        'F_NUMBER': 1.,
        'FOCAL_LENGTH': 2.9e-2,
        'MAX_DPT': 3.,
    }

def train():
    
    if args.device == "cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    data_loader,total_step = fs.load_data()
    train_data = data_loader[0]
    
    if args.continue_from:
        model = torch.load(args.continue_from).to(device)
    else:
        model = fu.FUNet(args.input_channels,args.output_channels,args.W,args.D).to(device)
        
    opt = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
    model.train()
    loss_e = 0
    print("---------------------start train---------------------")
    focal_length = args.focal_length
    f_number = args.fnumber
    for e in range(args.epochs):
        for i,batch in enumerate(train_data):

            image,depth,focus_dist = batch['input'].float().to(device),batch['output'].float().to(device),batch['focus_dist'].float().to(device)
            # print(image.shape)
            # print(depth.shape)
            # print(focus_dist.shape)
            # exit()
            # unloader(image[0][0].to('cpu')).save("sample.png")
            # unloader(depth[0][0].to('cpu')).save("sample_depth.png")
            
            # logits = model(image,focal_length,focus_dist,f_number)
            logits = model(image,focus_dist)
            # print(logits)
            # exit()
            B,H,W = logits.shape
            depth = depth.squeeze()
            
            loss_ssim = 1 - criterion['ssim'](logits.contiguous().view(B, 1, H, W), depth.contiguous().view(B, 1, H, W))
            loss_l1 = criterion['l1'](logits.contiguous().view(B, 1, H, W), depth.contiguous().view(B, 1, H, W))
            # mseloss = nn.MSELoss()
            # loss_l1 = mseloss(logits,depth)
            loss_sharp = criterion['sharp'](logits.contiguous().view(B, 1, H, W), depth.contiguous().view(B, 1, H, W))
            
            # loss_ssim = 1 - criterion['ssim'](logits.contiguous().view(B*FS, 1, H, W), depth.contiguous().view(B*FS, 1, H, W))
            # loss_l1 = criterion['l1'](logits.contiguous().view(B*FS, 1, H, W), depth.contiguous().view(B*FS, 1, H, W))
            # # mseloss = nn.MSELoss()
            # # loss_l1 = mseloss(logits,depth)
            # loss_sharp = criterion['sharp'](logits.contiguous().view(B*FS, 1, H, W), depth.contiguous().view(B*FS, 1, H, W))
            loss_recon = args.recon_loss_alpha * loss_ssim + (1 - args.recon_loss_alpha) * loss_l1
            loss_b = loss_recon * args.recon_loss_lambda + loss_sharp * args.sharp_loss_lambda
            
            print(loss_ssim," ",loss_l1," ",loss_sharp)
            
            loss_e += loss_b.item()
            opt.zero_grad()
            if loss_b.item() < 5:
                loss_b.backward()
            opt.step()
        

            print("epochs:",e+1,"  batchs:",i+1,"  loss:",loss_b.item())
        loss_e /= total_step
        print("epochs:",e+1," finished,average loss:",loss_e)
        dirc = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.mkdir(args.img_save_dir+'train/'+dirc)
        os.mkdir(args.model_dir+dirc)
        torch.save(model,args.model_dir+dirc+"/model_{0:4f}.bin".format(loss_e))
        # [torchvision.utils.save_image(image[0][j],args.img_save_dir+"train/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
        # torchvision.utils.save_image(depth[0][0].to('cpu'),args.img_save_dir+"train/"+str(dirc)+"/d_"+str(i)+".png")
        # [torchvision.utils.save_image(logits[0][j].to('cpu').detach(),args.img_save_dir+"train/"+str(dirc)+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[1])]
        # [unloader(image[0][j]).save(args.img_save_dir+"train/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
        # unloader(depth[0].to('cpu')).save(args.img_save_dir+"train/"+str(dirc)+"/d_"+str(i)+".png")
        # [unloader(logits[0][j].to('cpu').detach()).save(args.img_save_dir+"train/"+str(dirc)+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[1])]
        [unloader(image[0][j]).save(args.img_save_dir+"train/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
        unloader(depth[0].to('cpu')).save(args.img_save_dir+"train/"+str(dirc)+"/d_"+str(i)+".png")
        unloader(logits[0].to('cpu').detach()).save(args.img_save_dir+"train/"+str(dirc)+"/pd_"+str(i)+".png")

def eval():
    if args.device == "cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_loader, total_step = fs.load_data()
    eval_data = data_loader[1]

    if args.eval_from_load:
        model = torch.load(args.model_dir + "_0.089532.bin").to(device)
    else:
        model = fu.FUNet(args.input_channels, args.output_channels, args.W, args.D).to(device)

    opt = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
    model.eval()
    focal_length = args.focal_length
    f_number = args.fnumber
    loss_e = 0
    print("---------------------start eval---------------------")
    for i, batch in enumerate(eval_data):
        image, depth, focus_dist = batch['input'].float().to(device), batch['output'].float().to(device), batch['focus_dist'].float().to(device)
        logits = model(image,focal_length,focus_dist,f_number)
        B,FS,H,W = logits.shape
        depth = depth.expand_as(logits)
        
        loss_ssim = 1 - criterion['ssim'](logits.contiguous().view(B*FS, 1, H, W), depth.contiguous().view(B*FS, 1, H, W))
        loss_l1 = criterion['l1'](logits.contiguous().view(B*FS, 1, H, W), depth.contiguous().view(B*FS, 1, H, W))
        loss_sharp = criterion['sharp'](logits.contiguous().view(B*FS, 1, H, W), depth.contiguous().view(B*FS, 1, H, W))
        loss_recon = 0 * loss_ssim + 1 * loss_l1
        loss_b = loss_recon * 1 + loss_sharp * 0
        
        loss_e += loss_b.item()
        print("batchs:", i + 1, "  loss:", loss_b.item())
        if i % 20 == 0:
            image = image.squeeze(0)
            depth = depth.squeeze(0)
            logits = logits.squeeze(0)
            # print(depth[0])
            # print(logits[0])
            # print(torch.max(logits[0]))
            # print(torch.min(logits[0]))
            dirc = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            os.mkdir(args.img_save_dir+dirc)
            [torchvision.utils.save_image(image[j],args.img_save_dir+"eval"+dirc+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[0])]
            torchvision.utils.save_image(depth[0].to('cpu'),args.img_save_dir+"eval"+dirc+"/d_"+str(i)+".png")
            [torchvision.utils.save_image(logits[j].to('cpu'),args.img_save_dir+"eval"+dirc+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[0])]
            # print(np.array(image[0].to('cpu')).shape)
            # [imageio.imwrite(args.img_save_dir+"i_{0}_".format(j)+str(loss_b.item())+".png",np.array(image[j].to('cpu'))) for j in range(image.shape[0])]
            # imageio.imwrite(args.img_save_dir+"d_"+str(loss_b.item())+".png",np.array(depth[0].to('cpu')))
            # [imageio.imwrite(args.img_save_dir+"pd_{0}_".format(j)+str(loss_b.item())+".png",np.array(logits[j].to('cpu').detach())) for j in range(logits.shape[0])]
    loss_e /= total_step
    
    print("total_loss:", loss_e)
        
train()
# eval()  #undebug