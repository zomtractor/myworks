import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("..")
import torch
import cv2
from skimage import img_as_ubyte
import yaml
from utils import network_parameters
import torch.optim as optim
import time
import utils
import numpy as np
import random
import math
import torch.nn.functional as F
from DataPro.data import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.UFT import UFCNet
from utils import ASLloss,ColorLoss,Blur,L1_Charbonnier_loss,img_pad,calculate_metrics,SSIM_loss,VGGLoss
import lpips
import warnings
from lightning.fabric import Fabric
warnings.filterwarnings("ignore")

## Set Seeds
my_seed = 1234
torch.backends.cudnn.benchmark = True
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

#Define Fabric
fabric=Fabric(accelerator="cuda",precision="16-mixed")
fabric.launch()


## Load yaml configuration file
with open('config.yaml', 'r') as config:
    opt = yaml.safe_load(config)

Train = opt['TRAINING']
OPT = opt['TRAINOPTIM']

## Model
print('==> Build the model')
model_restored = UFCNet()
p_number = network_parameters(model_restored)
# model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)

train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']
utils.mkdir("./val_result")
utils.mkdir("./mask_val_syn")
## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'train_logs')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
model_restored,optimizer=fabric.setup(model_restored,optimizer)
## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                       eta_min=float(OPT['LR_MIN']))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    md = '/mnt/zjy/zjy_flare_removal/PackageFlareRemoval/other_works/zbl/best_test/checkpoints'
    model_name = 'best194.pth'
    path_chk_rest = utils.get_last_path(md, model_name)
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
# Als = ASLloss().cuda()
# cl = ColorLoss().cuda()
# blur_rgb = Blur(3).cuda()
SSIMloss=SSIM_loss().cuda()
Charloss = L1_Charbonnier_loss().cuda()
Vgg_loss=VGGLoss().cuda()
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=OPT['BATCH'], drop_last=True)
val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2,
                        drop_last=True)
train_loader=fabric.setup_dataloaders(train_loader)
val_loader=fabric.setup_dataloaders(val_loader)
# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'] + 1)}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}''')
print('------------------------------------------------------------------')


# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
best_lpips=1000
best_epoch_lpips=0

total_start_time = time.time()
loss_fn_alex = lpips.LPIPS(net='alex')
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_mask_loss = 0
    epoch_total_loss = 0
    train_id = 1

    model_restored.train()
    for i, data in enumerate(train_loader, 0):
        # Forward propagation
        # for param in model_restored.parameters():
        #     param.grad = None
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        flare_mask = data[2].cuda()
        masked,restored = model_restored(input_)

        # Compute loss
        charl1 = Charloss(restored, target)
        ssim_loss = (1 - SSIMloss(restored, target))
        # color_loss = cl(blur_rgb(restored), blur_rgb(target))
        vgg_loss = Vgg_loss(restored, target)
        # loss_total = charl1 + ssim_loss + 0.01 * vgg_loss  # Ours
        # loss_total = charl1 + ssim_loss + 0.1 * vgg_loss  # l2 = 0.1
        # loss_total = charl1 + ssim_loss + 0.05 * vgg_loss  # l2 = 0.05
        # loss_total = charl1 + ssim_loss + 0.002 * vgg_loss  # l2 = 0.002
        loss_total = charl1 + ssim_loss + 0.002 * vgg_loss  # l2 = 0.001

        # charl1_mask = Charloss(restored * (flare_mask), target * (flare_mask))
        # ssim_loss_mask = (1 - SSIMloss(restored * (flare_mask), target * (flare_mask)))
        # # color_loss = cl(blur_rgb(restored), blur_rgb(target))
        # vgg_loss_mask = Vgg_loss(restored * (flare_mask), target * (flare_mask))
        # loss_mask = charl1_mask + ssim_loss_mask + 0.01 * vgg_loss_mask  # 损失函数
        loss_mask = structure_loss(masked, 1-flare_mask)

        loss = loss_total + 0.01 * loss_mask    # ours
        # loss = loss_total + 0.1 * loss_mask    #l1 = 0.1
        # loss = loss_total + 0.05 * loss_mask    #l1 = 0.05
        # loss = loss_total + 0.002 * loss_mask    #l1 = 0.002
        # loss = loss_total + 0.001 * loss_mask    #l1 = 0.001
        fabric.backward(loss)
        optimizer.step()
        epoch_mask_loss += 0.01 * loss_mask.item()
        epoch_loss += loss.item()
        epoch_total_loss += loss_total.item()
    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        # psnr_val_rgb = []
        # ssim_val_rgb = []
        cumulative_psnr = 0
        cumulative_ssim = 0
        cumulative_lpips = 0
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            b, c, h, w = input_.size()
            k = 16
            # pad image such that the resolution is a multiple of 32
            w_pad = (math.ceil(w / k) * k - w) // 2
            h_pad = (math.ceil(h / k) * k - h) // 2
            w_odd_pad = w_pad
            h_odd_pad = h_pad
            if w % 2 == 1:
                w_odd_pad += 1
            if h % 2 == 1:
                h_odd_pad += 1
            input_ = img_pad(input_, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
            #input_ = imageprocess.addpadding(32,input_)

            with torch.no_grad():
                masked,restored = model_restored(input_)
            # for res, tar in zip(restored, target):
            #     psnr_val_rgb.append(utils.torchPSNR(res, tar))
            #     ssim_val_rgb.append(utils.torchSSIM(restored, target))
                if h_pad != 0:
                    restored = restored[:, :, h_pad:-h_odd_pad, :]
                if w_pad != 0:
                    restored = restored[:, :, :, w_pad:-w_odd_pad]
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                cv2.imwrite(os.path.join('./val_result', data_val[2][batch] + '.png'),cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
                result = cv2.imread(os.path.join('./val_result', data_val[2][batch] + '.png'))
                gt = cv2.imread(os.path.join(val_dir+'/gt', data_val[2][batch]) + '.png')


                # result = np.array(result, dtype = 'uint8')
                cur_lpips, cur_psnr, cur_ssim = calculate_metrics(result, gt,loss_fn_alex)
                cumulative_psnr += cur_psnr
                cumulative_ssim += cur_ssim
                cumulative_lpips += cur_lpips
            masked = masked.sigmoid().data.cpu().numpy().squeeze()
            masked = (masked - masked.min()) / (masked.max() - masked.min() + 1e-8)
            cv2.imwrite(os.path.join('./mask_val_syn', data_val[2][0] + '.png'), masked * 255)
        # psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        # ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
        psnr_val_rgb = cumulative_psnr / len(val_loader) # actually YCbCr
        ssim_val_rgb = cumulative_ssim / len(val_loader)
        lpips_val_rgb = cumulative_lpips / len(val_loader)


        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        # Save the best LPIPS model of validation
        if lpips_val_rgb < best_lpips:
            best_lpips = lpips_val_rgb
            best_epoch_lpips = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestLPIPS.pth"))
        print("[epoch %d LPIPS: %.4f --- best_epoch %d Best_LPIPS %.4f]" % (
            epoch, lpips_val_rgb, best_epoch_lpips, best_lpips))

        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))


        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
        writer.add_scalar('val/LPIPS', lpips_val_rgb, epoch)

    scheduler.step()

    print("------------------------------------------------------------------")
    print(
        "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLossMask: {:.4f}\tLossTotal: {:.4f}\tLearningRate {:.8f}".format(epoch,
                                                                                                                 time.time() - epoch_start_time,
                                                                                                                 epoch_loss,
                                                                                                                 epoch_mask_loss,
                                                                                                                 epoch_total_loss,
                                                                                                                 scheduler.get_lr()[
                                                                                                                     0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/base_loss', epoch_total_loss, epoch)
    writer.add_scalar('train/flare_loss',epoch_mask_loss, epoch)
    # writer.add_scalar('train/vgg_loss', epoch_loss-epoch_ssim_loss-epoch_c1_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))




