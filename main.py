import torch
import torch.nn as nn
import os
import FUNet as fu
from utils import *
# import datasets as ds
import fs
import torchvision
import numpy as np
import cv2
import imageio
from datetime import datetime
import torch.nn.functional as F
# from loss import BlurMetric
from PIL import Image
import cal_Fda as fda
import math
import time

parser = get_parser()
args = parser.parse_args()
unloader = torchvision.transforms.ToPILImage()

def train():
    
    if args.device == "cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # data_loader,total_step = fs.load_data()
    # train_data = data_loader[0]
    
    dataset_config = get_data_config(args)
    dataloaders = load_data(dataset_config, args.dataset, args.BS)
    train_data = dataloaders[0]

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
    camera = get_camera(args)
    focal_length = camera.focal_length
    f_number = camera.fnumber
    pixel_size = camera.pixel_size
    # focal_length = args.focal_length
    # f_number = args.fnumber
    lowest_loss = 10.
    for e in range(args.epochs):
        for i,batch in enumerate(train_data):

            # image,depth,focus_dist = batch['input'].float().to(device),batch['output'].float().to(device),batch['focus_dist'].float().to(device)
            image,depth,focus_dist = batch['rgb_fd'].float().to(device),batch['dpt'].float().to(device),batch['rgb_fd'][:,:,-1][:,:,0,0].float().to(device)
            depth = F.avg_pool2d(depth, kernel_size=2, stride=2, padding=0)
            # B,FS,C,H,W = image.shape
            # image = torch.cat([image,focus_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand([B,FS,1,H,W])],2)
            # print(image.shape)
            # print(depth.shape)
            # print(focus_dist.shape)
            # print(image.shape)
            # print(depth.shape)
            # print(focus_dist.shape)
            # exit()
            # unloader(image[0][0].to('cpu')).save("sample.png")
            # unloader(depth[0][0].to('cpu')).save("sample_depth.png")
            
            # logits,cstack_loss_ssim,cstack_loss_l1 = model(image,focal_length,focus_dist,f_number)
            
            # logits,cstack_loss_ssim,cstack_loss_l1 = model(image,focal_length,focus_dist,f_number,pixel_size)

            logits,cstack_loss_l1 = model(image,focal_length,f_number,pixel_size)

            # logits = model(image,focal_length)
            
            # print(logits)
            # exit()
            
            B,C,H,W = logits.shape
            
            depth = depth.expand_as(logits)
            
            logits, depth = logits.view(B*C,1,H,W), depth.contiguous().view(B*C,1,H,W)
            
            # depth = F.avg_pool2d(depth, kernel_size=2, stride=2, padding=0)
            
            # depth = depth.squeeze()
            
            
            # sigma = fda.cal_sigma(depth,focal_length,focus_dist,f_number,pixel_size)
            
            # loss_ssim = 1 - criterion['ssim'](logits.contiguous().view(B, 1, H, W), depth.contiguous().view(B, 1, H, W))
            # loss_l1 = criterion['l1'](logits.contiguous().view(B, 1, H, W), depth.contiguous().view(B, 1, H, W))
            # # mseloss = nn.MSELoss()
            # # loss_l1 = mseloss(logits,depth)
            # loss_sharp = criterion['sharp'](logits.contiguous().view(B, 1, H, W), depth.contiguous().view(B, 1, H, W))
            
            # loss_ssim = 1 - criterion['ssim'](logits.contiguous().view(B*FS, C, H, W), sigma.contiguous().view(B*FS, C, H, W))
            # loss_l1 = criterion['l1'](logits.contiguous().view(B*FS, C, H, W), sigma.contiguous().view(B*FS, C, H, W))
            # # mseloss = nn.MSELoss()
            # # loss_l1 = mseloss(logits,depth)
            # loss_sharp = criterion['sharp'](logits.contiguous().view(B*FS, C, H, W), sigma.contiguous().view(B*FS, C, H, W))
            
            loss_ssim = 1 - criterion['ssim'](logits, depth)
            loss_l1 = cstack_loss_l1 * args.cstack_loss_beta + criterion['l1'](logits, depth) * (1 - args.cstack_loss_beta)
            
            # loss_ssim = args.cstack_loss_beta*cstack_loss_ssim + (1-args.cstack_loss_beta)*depth_loss_ssim
            # loss_l1 = args.cstack_loss_beta*cstack_loss_l1 + (1-args.cstack_loss_beta)*depth_loss_l1
            # mseloss = nn.MSELoss()
            # loss_l1 = mseloss(logits,depth)
            loss_blur = criterion['blur'](logits)
            loss_sharp = criterion['sharp'](logits, depth)
            
            loss_recon = args.recon_loss_alpha * loss_ssim + (1 - args.recon_loss_alpha) * loss_l1
            loss_b = loss_recon * args.recon_loss_lambda + loss_sharp * args.sharp_loss_lambda + loss_blur * args.blur_loss_lambda
            
            # print(loss_ssim," ",loss_l1," ",loss_sharp," ",loss_blur)
            
            loss_e += loss_b.item()
            if math.isnan(loss_b.item()):
                print(logits)
                print("loss nan!")
                exit()
            opt.zero_grad()
            # if loss_b.item() < 5:
            loss_b.backward()
            opt.step()
        

            print("epochs:",e+1,"  batchs:",i+1,"  loss:",loss_b.item())
        loss_e /= i+1
        if loss_e < lowest_loss:
            lowest_loss = loss_e
            print("epochs:",e+1," finished,average loss:",loss_e)
            dirc = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            os.mkdir(args.img_save_dir+'train_cstack/'+dirc)
            os.mkdir(args.model_dir+dirc+"_cstack")
            torch.save(model.state_dict(),args.model_dir+dirc+"_cstack"+"/model_{0:4f}.bin".format(loss_e)) 
        
        # rgb_d = fda.cal_d(logits[0],focal_length,focus_dist[0],f_number,pixel_size)
        # pre_d = torch.sum(fda.cmp(rgb_d[:,0],rgb_d[:,1],rgb_d[:,2]),1)/3
        
        # [torchvision.utils.save_image(image[0][j],args.img_save_dir+"train/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
        # torchvision.utils.save_image(depth[0][0].to('cpu'),args.img_save_dir+"train/"+str(dirc)+"/d_"+str(i)+".png")
        # [torchvision.utils.save_image(logits[0][j].to('cpu').detach(),args.img_save_dir+"train/"+str(dirc)+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[1])]
            image = image[:,:,:-1]
            [unloader(image[0][j]).save(args.img_save_dir+"train_cstack/"+dirc+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
            unloader(depth[0][0].to('cpu')).save(args.img_save_dir+"train_cstack/"+dirc+"/d_"+str(i)+".png")
            [unloader(logits[0][j].to('cpu').detach()).save(args.img_save_dir+"train_cstack/"+dirc+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[1])]
       
        # [unloader(image[0][j]).save(args.img_save_dir+"train/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
        # unloader(depth[0][0].to('cpu')).save(args.img_save_dir+"train/"+str(dirc)+"/d_"+str(i)+".png")
        # [unloader(pre_d[j].to('cpu').detach()).save(args.img_save_dir+"train/"+str(dirc)+"/pd_{0}_".format(j)+str(i)+".png") for j in range(pre_d.shape[0])]
        
        # [unloader(image[0][j]).save(args.img_save_dir+"train/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
        # unloader(depth[0].to('cpu')).save(args.img_save_dir+"train/"+str(dirc)+"/d_"+str(i)+".png")
        # unloader(logits[0].to('cpu').detach()).save(args.img_save_dir+"train/"+str(dirc)+"/pd_"+str(i)+".png")

def eval():
    if args.device == "cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_loader, total_step = fs.load_data()
    eval_data = data_loader[1]
    
    # dataset_config = get_data_config(args)
    # dataloaders = load_data(dataset_config, args.dataset, args.BS)
    # eval_data = dataloaders[1]
    
    model = fu.FUNet(args.input_channels, args.output_channels, args.W, args.D).to(device)
    if args.eval_from_load:
        model.load_state_dict(torch.load(args.eval_from_load))
        # model = torch.load(args.eval_from_load)

    opt = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
    model.eval()
    # camera = get_camera(args)
    # focal_length = camera.focal_length
    # f_number = camera.fnumber
    # pixel_size = camera.pixel_size
    focal_length = args.focal_length
    f_number = args.fnumber
    loss_e = 0
    total_ssim = 0
    total_l1 = 0
    total_blur = 0
    total_sharp = 0
    total_abs = 0
    total_sqr = 0
    total_rmse = 0
    total_rml = 0
    total_d1 = 0
    total_d2 = 0
    total_d3 = 0
    print("---------------------start eval---------------------")
    for i, batch in enumerate(eval_data):
        image,depth,focus_dist = batch['input'].float().to(device),batch['output'].float().to(device),batch['focus_dist'].float().to(device)
        # image,depth,focus_dist = batch['rgb_fd'].float().to(device),batch['dpt'].float().to(device),batch['rgb_fd'][:,:,-1][:,:,0,0].float().to(device)
        B,FS,C,H,W = image.shape
        # depth = F.avg_pool2d(depth, kernel_size=2, stride=2, padding=0)
        image = torch.cat([image,focus_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand([B,FS,1,H,W])],2)
        # image, depth, focus_dist = batch['rgb_fd'][:,:,:-1].float().to(device), batch['dpt'].float().to(device), batch['rgb_fd'][:,:,-1].float().to(device)
        # logits = model(image,focal_length,focus_dist,f_number)
        logits = model(image)
        B,FS,H,W = logits.shape
        depth = depth.expand_as(logits)
            
        logits, depth = logits.view(B*FS,1,H,W), depth.contiguous().view(B*FS,1,H,W)
        AbsRel, SqRel, RMSE, RMSE_log, delta1, delta2, delta3 = eval_depth(logits, depth)
        
        loss_ssim = 1 - criterion['ssim'](logits, depth)
        loss_l1 = criterion['l1'](logits, depth)
        loss_sharp = criterion['sharp'](logits, depth)
        loss_blur = criterion['blur'](logits)
            
        loss_recon = args.recon_loss_alpha * loss_ssim + (1 - args.recon_loss_alpha) * loss_l1
        loss_b = loss_recon * args.recon_loss_lambda + loss_sharp * args.sharp_loss_lambda + loss_blur * args.blur_loss_lambda
        
        print(loss_ssim," ",loss_l1," ",loss_sharp," ",loss_blur)
        
        loss_e += loss_b.item()
        total_ssim += 1-loss_ssim.item()
        total_l1 += loss_l1.item()
        total_blur += loss_blur.item()
        total_sharp += loss_sharp.item()
        total_abs += AbsRel.item()
        total_sqr += SqRel.item()
        total_rmse += RMSE.item()
        total_rml += RMSE_log.item()
        total_d1 += delta1.item()
        total_d2 += delta2.item()
        total_d3 += delta3.item()
        print("batchs:", i + 1, "  loss:", loss_b.item())
        if i % 5 == 0:
            time.sleep(1)
            # image = image.squeeze(0)
            # depth = depth.squeeze(0)
            # logits = logits.squeeze(0)
            # print(depth[0])
            # print(logits[0])
            # print(torch.max(logits[0]))
            # print(torch.min(logits[0]))
            dirc = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            os.mkdir(args.img_save_dir+"eval/"+dirc)
            # [torchvision.utils.save_image(image[j],args.img_save_dir+"eval"+dirc+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[0])]
            # torchvision.utils.save_image(depth[0].to('cpu'),args.img_save_dir+"eval"+dirc+"/d_"+str(i)+".png")
            # [torchvision.utils.save_image(logits[j].to('cpu'),args.img_save_dir+"eval"+dirc+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[0])]
            # print(np.array(image[0].to('cpu')).shape)
            # [imageio.imwrite(args.img_save_dir+"i_{0}_".format(j)+str(loss_b.item())+".png",np.array(image[j].to('cpu'))) for j in range(image.shape[0])]
            # imageio.imwrite(args.img_save_dir+"d_"+str(loss_b.item())+".png",np.array(depth[0].to('cpu')))
            # [imageio.imwrite(args.img_save_dir+"pd_{0}_".format(j)+str(loss_b.item())+".png",np.array(logits[j].to('cpu').detach())) for j in range(logits.shape[0])]
            image = image[:,:,:-1]
            [unloader(image[0][j]).save(args.img_save_dir+"eval/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
            unloader(((depth[0][0]-torch.min(depth[0][0]))/(torch.max(depth[0][0])-torch.min(depth[0][0]))).to('cpu')).save(args.img_save_dir+"eval/"+str(dirc)+"/d_"+str(i)+".png")
            [unloader(((logits[0][j]-torch.min(logits[0][j]))/(torch.max(logits[0][j])-torch.min(logits[0][j]))).to('cpu').detach()).save(args.img_save_dir+"eval/"+str(dirc)+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[1])]
    loss_e /= (i+1)
    total_ssim /= (i+1)
    total_l1 /= (i+1)
    total_blur /= (i+1)
    total_sharp /= (i+1)
    total_abs /= (i+1)
    total_sqr /= (i+1)
    total_rmse /= (i+1)
    total_rml /= (i+1)
    total_d1 /= (i+1)
    total_d2 /= (i+1)
    total_d3 /= (i+1)

    print("total_loss:", loss_e,"total_ssim:", total_ssim,"total_l1:", total_l1,"total_blur:", total_blur,"total_sharp:", total_sharp,"total_abs:", total_abs,"total_sqr:", total_sqr,"total_rmse:", total_rmse,"total_rml:", total_rml,"total_d1:", total_d1,"total_d2:", total_d2,"total_d3:", total_d3)
        
train()
# eval()  #undebug