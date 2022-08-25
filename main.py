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
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from apex import amp
from apex.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler,autocast
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0,1,2,3'

unloader = torchvision.transforms.ToPILImage()


def train(rank,ranks,args):
    
    # dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=rank)
    # torch.cuda.set_device(rank)
    
    if rank == 0:
        wandb.init(project="dfd", dir='..', name=args.name)
        wandb.config.update(args)
    
    if args.device == "cuda":
        device = torch.device('cuda:'+str(rank))
    else:
        device = torch.device('cpu')
        
    # data_loader,total_step = fs.load_data()
    # train_data = data_loader[0]
    
    dataset_config = get_data_config(args)
    dataloaders = load_data(dataset_config, args.dataset, args.BS, rank)
    train_data = dataloaders[0]
    

    if args.continue_from:
        model = torch.load(args.continue_from).to(device)
    else:
        model = fu.FUNet(args.input_channels,args.output_channels,args.W,args.D).to(device)
        
    # torch.cuda.set_device(rank)
    
    
    opt = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    
    # model, opt = amp.initialize(model, opt, opt_level="O1")
    # model = DistributedDataParallel(model)
    # torch.distributed
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],output_device=rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # scaler = GradScaler()
    
    criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
    model.train()
    loss_e = 0
    loss_l1_e = 0
    pred_loss_l1_e = 0
    cstack_loss_l1_e = 0
    loss_ssim_e = 0
    loss_blur_e = 0
    loss_sharp_e = 0
    if rank == 0:
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

            # sigma,sigma_gt,logits,cstack_loss_l1 = model(image,focal_length,f_number,pixel_size,depth)
            
            # with autocast():
            logits = model(image,focal_length)
            
            # print(logits)
            # exit()
            
            B,C,H,W = logits.shape
            
            pred = logits
            gt = depth.expand_as(pred)
            
            pred, gt = pred.view(B*C,1,H,W), gt.contiguous().view(B*C,1,H,W)
            
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
            
            loss_ssim = 1 - criterion['ssim'](pred, gt)
            loss_l1 = criterion['l1'](pred, gt)
            # loss_l1 = cstack_loss_l1 * args.cstack_loss_beta + pred_loss_l1 * (1 - args.cstack_loss_beta)
            
            # loss_ssim = args.cstack_loss_beta*cstack_loss_ssim + (1-args.cstack_loss_beta)*depth_loss_ssim
            # loss_l1 = args.cstack_loss_beta*cstack_loss_l1 + (1-args.cstack_loss_beta)*depth_loss_l1
            # mseloss = nn.MSELoss()
            # loss_l1 = mseloss(logits,depth)
            loss_blur = criterion['blur'](pred)
            loss_sharp = criterion['sharp'](pred, gt)
            
            loss_recon = args.recon_loss_alpha * loss_ssim + (1 - args.recon_loss_alpha) * loss_l1
            loss_b = loss_recon * args.recon_loss_lambda + loss_sharp * args.sharp_loss_lambda + loss_blur * args.blur_loss_lambda
            
            # print(loss_ssim," ",loss_l1," ",loss_sharp," ",loss_blur)
            
            loss_e += loss_b.item()
            # pred_loss_l1_e += pred_loss_l1.item()
            # cstack_loss_l1_e += cstack_loss_l1.item()
            loss_l1_e += loss_l1.item()
            loss_ssim_e += loss_ssim.item()
            loss_blur_e += loss_blur.item()
            loss_sharp_e += loss_sharp.item()
            if math.isnan(loss_b.item()):
                print(pred)
                print("loss nan!")
                exit()
            
            # if loss_b.item() < 5:
            # loss_b.backward()
            
            # opt.zero_grad()
            # scaler.scale(loss_b).backward()
            # scaler.step(opt)
            # scaler.update()
            
            opt.zero_grad()
            loss_b.backward()
            opt.step()
            
            # opt.zero_grad()
            # with amp.scale_loss(loss_b, opt) as scaled_loss:
            #     scaled_loss.backward()
            # opt.step()
        
            if rank == 0:
                print("epochs:",e+1,"  batchs:",i+1,"  loss:",loss_b.item())
        loss_e /= i+1
        # pred_loss_l1_e /= i+1
        # cstack_loss_l1_e /= i+1
        loss_l1_e /= i+1
        loss_ssim_e /= i+1
        loss_blur_e /= i+1
        loss_sharp_e /= i+1
        if rank == 0:
            print("epochs:",e+1," finished,average loss:",loss_e)
            if loss_e < lowest_loss:
                lowest_loss = loss_e
                
                dirc = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                os.mkdir(args.img_save_dir+'train/'+dirc)
                os.mkdir(args.model_dir+dirc+"")
                torch.save(model.state_dict(),args.model_dir+dirc+""+"/model_{0:4f}.bin".format(loss_e)) 
                
                image = image[:,:,:-1]
                [unloader(image[0][j]).save(args.img_save_dir+"train/"+dirc+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
                unloader((depth[0][0]/args.camera_far).to('cpu')).save(args.img_save_dir+"train/"+dirc+"/d_"+str(i)+".png")
                [unloader((logits[0][j].float()/args.camera_far).to('cpu').detach()).save(args.img_save_dir+"train/"+dirc+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[1])]
        
        if rank == 0:
            logs = dict(loss_l1=loss_l1_e,
                        # pred_loss_l1=pred_loss_l1_e,
                        # cstack_loss_l1=cstack_loss_l1_e,
                        loss_ssim=loss_ssim_e,
                        loss_blur=loss_blur_e,
                        loss_sharp=loss_sharp_e,
                        loss=loss_e,
                        logits=wandb.Image(logits[0][0].float().to('cpu').detach()/args.camera_far),
                        depth=wandb.Image(depth[0][0].to('cpu')),
                        image=wandb.Image(image[0][0].to('cpu')/args.camera_far))
            wandb.log({"train":logs})
        # rgb_d = fda.cal_d(logits[0],focal_length,focus_dist[0],f_number,pixel_size)
        # pre_d = torch.sum(fda.cmp(rgb_d[:,0],rgb_d[:,1],rgb_d[:,2]),1)/3
        
        # [torchvision.utils.save_image(image[0][j],args.img_save_dir+"train/"+str(dirc)+"/i_{0}_".format(j)+str(i)+".png") for j in range(image.shape[1])]
        # torchvision.utils.save_image(depth[0][0].to('cpu'),args.img_save_dir+"train/"+str(dirc)+"/d_"+str(i)+".png")
        # [torchvision.utils.save_image(logits[0][j].to('cpu').detach(),args.img_save_dir+"train/"+str(dirc)+"/pd_{0}_".format(j)+str(i)+".png") for j in range(logits.shape[1])]
       
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

if __name__ == "__main__":
    train(0,4,args)
    # mp.spawn(train, nprocs=4,args=(4,args,))
    # eval()  #undebug