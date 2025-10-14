from difflib import restore
import random
import sys
import os
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("..")
import utils
from utils import *
from lightning.fabric import Fabric
from Models.model0601_wo_net_dwt import MNet
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from utils.utils import L1_Charbonnier_loss, img_pad, calculate_metrics, SSIM_loss, VGGLoss
from DataPro.data import get_training_data, get_validation_data
from utils.feature_visualization import draw_feature_map
from torch.utils.data import DataLoader
import time
import lpips
import math
from skimage import img_as_ubyte
from torch.nn import functional as F

# set seeds
torch.backends.cudnn.benchmark = True  # 开启benchmark模式后，PyTorch会尝试自动找到最佳的算法来运行你的模型
my_seed = 1234
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)  # torch 库的随机种子
torch.cuda.manual_seed_all(my_seed)  # PyTorch在CUDA环境中所有GPU上的随机数生成器的种子

# Define Fabric
fabric = Fabric(accelerator="cuda", precision="16-mixed")
fabric.launch()

# load yaml
with open(file="../configs/config0601_wo_net_dwt.yaml", mode="r") as config:
    opt = yaml.safe_load(config)

TRAIN = opt["TRAINING"]
OPT = opt["TRAINOPTIM"]

# load model
model_base = MNet(dim=32, num_blocks=[1, 2, 8, 2])
# model_base = Uformer(img_size=256, embed_dim=32, win_size=8, token_mlp='leff',
#                      depths=[1, 2, 8, 8, 4, 8, 8, 2, 1], dd_in=3, batch_size=OPT['BATCH'])
base_parameters_number = network_parameters(model_base)
print("Net parameters: {}".format(base_parameters_number))
model_base.cuda()

# model and dataset path
model_name = opt['MODEL']['NAME']
train_batch = opt['MODEL']['TRAIN_BATCH']
resume_batch = opt['MODEL']['RESUME_BATCH']
resume_name = model_name + "_" + resume_batch
model_name = model_name + "_" + train_batch
print(model_name)
model_path = os.path.join(TRAIN["SAVE_DIR"], model_name)
resume_path = os.path.join(TRAIN["SAVE_DIR"], resume_name)
utils.mkdir(model_path)
print(model_path)

train_dir = TRAIN['TRAIN_DIR']
val_dir = TRAIN['VAL_DIR']

# log save path
log_dir = os.path.join(TRAIN["LOG_DIR"], model_name)
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{model_name}')

# optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_base.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
model_base, optimizer = fabric.setup(model_base, optimizer)

# scheduler strategy
warmup_epochs = OPT['WARMUP_EPOCHS']
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))  # 余弦退火方式调整优化器
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# resume (if its necessary)
if TRAIN['RESUME']:
    path_chk_rest = utils.get_last_path(resume_path, '_bestPSNR.pth')
    utils.load_checkpoint(model_base, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

# loss set
SSIMLoss = SSIM_loss().cuda()
CharLoss = L1_Charbonnier_loss().cuda()
VGGLoss = VGGLoss().cuda()
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

# dataloader
print('==> Loading Datasets')
train_dataset = get_training_data(train_dir, {'patch_size': TRAIN['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=OPT['BATCH'], drop_last=True)
val_dataset = get_validation_data(val_dir, {'patch_size': TRAIN['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2,
                        drop_last=True)
train_loader = fabric.setup_dataloaders(train_loader)
val_loader = fabric.setup_dataloaders(val_loader)

# show train config
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {model_name}
    Train patches size: {str(TRAIN['TRAIN_PS']) + 'x' + str(TRAIN['TRAIN_PS'])}
    Val patches size:   {str(TRAIN['VAL_PS']) + 'x' + str(TRAIN['VAL_PS'])}
    Model parameters:   {base_parameters_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'] + 1)}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}''')
print('------------------------------------------------------------------')
# start training
best_psnr = 0
best_ssim = 0
best_psnr_epoch = 0
best_ssim_epoch = 0
best_lpips = 1000
best_lpips_epoch = 0

total_start_time = time.time()
loss_fn_alex = lpips.LPIPS(net='alex')
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    # 设置loss 可以用多个loss
    epoch_loss = 0
    epoch_total_loss = 0

    model_base.train()
    print("train loader length: {}", len(train_loader))
    # 前向传播
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        gt = data[0].cuda()
        input_ = data[1].cuda()
        # mask = data[2].cuda()
        # source = data[5].cuda()
        # source_mask = data[6].cuda()

        restored = model_base(input_)
        name = data[3]


        # 正常计算loss
        char_l1_loss = CharLoss(restored, gt)
        ssim_loss = (1 - SSIMLoss(restored, gt))
        vgg_loss = VGGLoss(restored, gt)
        image_loss = char_l1_loss + ssim_loss + 0.01 * vgg_loss



        fabric.backward(image_loss)
        optimizer.step()

        epoch_total_loss += image_loss.item()

        

    # 验证
    if epoch % TRAIN['VAL_AFTER_EVERY'] == 0:
        model_base.eval()
        cumulative_psnr = 0
        cumulative_ssim = 0
        cumulative_lpips = 0
        for ii, data_val in enumerate(val_loader, 0):
            gt = data_val[0].cuda()
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
            with torch.no_grad():
                restored = model_base(input_)
                if h_pad != 0:
                    restored = restored[:, :, h_pad:-h_odd_pad, :]
                if w_pad != 0:
                    restored = restored[:, :, :, w_pad:-w_odd_pad]
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                temp_path = os.path.join(TRAIN['TRAIN_RESULT'], model_name)
                os.makedirs(temp_path, exist_ok=True)
                cv2.imwrite(os.path.join(temp_path, data_val[2][batch] + '.png'),
                            cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
                output = cv2.imread(os.path.join(temp_path, data_val[2][batch] + '.png'))
                gt = cv2.imread(os.path.join(val_dir, './gt', data_val[2][batch]) + '.png')

                cur_lpips, cur_psnr, cur_ssim = calculate_metrics(output, gt, loss_fn_alex)
                cumulative_psnr += cur_psnr
                cumulative_ssim += cur_ssim
                cumulative_lpips += cur_lpips
        psnr_val_rgb = cumulative_psnr / len(val_loader)
        ssim_val_rgb = cumulative_ssim / len(val_loader)
        lpips_val_rgb = cumulative_lpips / len(val_loader)

        # 保存验证集psnr表现最好的模型
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_psnr_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_base.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(model_path, "model_bestPSNR.pth"))
        print(("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_psnr_epoch, best_psnr)))

        # 保存验证集SSIM表现最好的模型
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_ssim_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_base.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_path, "model_bestSSIM_epoch.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_ssim_epoch, best_ssim))

        # 保存验证集LPIPS表现最好的模型
        if lpips_val_rgb < best_lpips:
            best_lpips = lpips_val_rgb
            best_lpips_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_base.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_path, "model_bestLPIPS.pth"))
        print("[epoch %d LPIPS: %.4f --- best_epoch %d Best_LPIPS %.4f]" % (
            epoch, lpips_val_rgb, best_lpips_epoch, best_lpips))

        """ 
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_path, f"model_epoch_{epoch}.pth"))
        """

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
        writer.add_scalar('val/LPIPS', lpips_val_rgb, epoch)

    scheduler.step()

    print("------------------------------------------------------------------")
    print(
        "Epoch: {}\tTime: {:.4f}\tImage Loss: {:.4f}\t LossTotal: {:.4f}\tLearningRate {:.8f}"
        .format(epoch, time.time() - epoch_start_time, image_loss, epoch_total_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_base.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_path, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/base_loss', epoch_total_loss, epoch)
    # writer.add_scalar('train/flare_loss', epoch_mask_loss, epoch)
    # writer.add_scalar('train/vgg_loss', epoch_loss-epoch_ssim_loss-epoch_c1_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
