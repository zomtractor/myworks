import os
import datetime

import torch
import cv2
from skimage import img_as_ubyte
from focal_frequency_loss import FocalFrequencyLoss as FFL
import yaml

import model
from model import CombinedLoss
from utils import network_parameters
import torch.optim as optim
import time
import utils
import numpy as np
import random
import math
from data import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.mask_utils import calculate_metrics
import lpips
import warnings
from lightning.fabric import Fabric

import torch.nn.functional as F

def init_torch_config(config):
    warnings.filterwarnings("ignore")
    # torch.set_float32_matmul_precision('high')
    ## Set Seeds
    my_seed = 1234
    torch.backends.cudnn.benchmark = True
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.set_float32_matmul_precision('high')
    #torch.set_anomaly_enabled(True)
    # fabric = Fabric(accelerator="cuda", devices=2, strategy="ddp_find_unused_parameters_true")
    fabric = Fabric(accelerator="cuda",devices=config['TRAINOPTIM']['DEVICES'],strategy="ddp_find_unused_parameters_true")
    fabric.launch()
    return fabric


def get_data_loaders(config, fabric):
    Train = config['TRAINING']
    OPT = config['TRAINOPTIM']
    ## DataLoaders
    print('==> Loading datasets')
    utils.mkdir(Train['VAL']['REAL_SAVE'])
    utils.mkdir(Train['VAL']['SYN_SAVE'])

    train_dataset = get_training_data(config['DATASET'], Train['TRAIN_PS'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                              shuffle=True, num_workers=OPT['BATCH'], drop_last=True)
    real_val_dataset = get_validation_data(Train['VAL']['REAL_DIR'], {'patch_size': Train['VAL_PS']})
    real_val_loader = DataLoader(dataset=real_val_dataset, batch_size=1, shuffle=False, num_workers=2,
                                 drop_last=True)
    syn_val_dataset = get_validation_data(Train['VAL']['SYN_DIR'], {'patch_size': Train['VAL_PS']})
    syn_val_loader = DataLoader(dataset=syn_val_dataset, batch_size=1, shuffle=False, num_workers=2,
                                drop_last=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    # real_val_loader = fabric.setup_dataloaders(real_val_loader)
    # syn_val_loader = fabric.setup_dataloaders(syn_val_loader)
    return train_loader, real_val_loader, syn_val_loader

def load_model(config, fabric):
    Train = config['TRAINING']
    OPT = config['TRAINOPTIM']

    print('==> Build the model')
    ## Training model path direction
    mode = config['MODEL']['MODE']
    model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
    utils.mkdir(model_dir)
    # model_restored = UBlock(base_channels=OPT['CHANNELS'])
    # model_restored = Uformer(embed_dim=10)
    model_class = getattr(model, config['MODEL']['ARCH'])
    model_args = config['MODEL']['ARGS']
    model_restored = model_class(**model_args)
    p_number = network_parameters(model_restored)
    ## Optimizer
    start_epoch = 1
    new_lr = float(OPT['LR_INITIAL'])
    optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    model_restored, optimizer = fabric.setup(model_restored, optimizer)
    ## Scheduler (Strategy)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                            eta_min=float(OPT['LR_MIN']))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
    checkpoint = None
    ## Resume (Continue training by a pretrained model)
    if Train['RESUME']:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        checkpoint = utils.load_checkpoint(model_restored, path_chk_rest)
        del path_chk_rest
        if (checkpoint is not None):
            # start_epoch = utils.load_start_epoch(path_chk_rest) + 1
            start_epoch = checkpoint['epoch'] + 1
            # utils.load_optim(optimizer, path_chk_rest)
            optimizer.load_state_dict(checkpoint['optimizer'])

            for i in range(1, start_epoch):
                scheduler.step()
            new_lr = scheduler.get_lr()[0]
            print('------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            print('------------------------------------------------------------------')
        else:
            print('No checkpoint found, starting from scratch.')
    # Show the training configuration
    print(f'''==> Training details:
        ------------------------------------------------------------------
            Restoration mode:   {mode}
            Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
            Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
            Model parameters:   \\{p_number}
            Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'] + 1)}
            Batch sizes:        {OPT['BATCH']}
            Learning rate:      {OPT['LR_INITIAL']}''')
    print('------------------------------------------------------------------')
    return model_restored, checkpoint, optimizer, scheduler, start_epoch

def assertLimited():
    start_time = datetime.time(7, 00)  # 7:40
    end_time = datetime.time(8, 15)    # 8:10
    assert not (start_time <= datetime.datetime.now().time() <= end_time), "当前时间位于禁止时间段 7:40~8:10 内"
    print("assert passed")
def load_config():
    ## Load yaml configuration file
    opt = None
    with open('config.yaml', 'r') as config:
        opt = yaml.safe_load(config)

    Train = opt['TRAINING']
    OPT = opt['TRAINOPTIM']
    mode = opt['MODEL']['MODE']

    ## Log
    log_dir = os.path.join(Train['SAVE_DIR'], mode, 'train_logs')
    utils.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')
    return opt, writer

def precompute_padding(h, w, k=16):
    w_pad = math.ceil(w / k) * k - w
    h_pad = math.ceil(h / k) * k - h
    w_pad_left = w_pad // 2
    h_pad_top = h_pad // 2
    w_pad_right = w_pad_left + (w_pad % 2)
    h_pad_bottom = h_pad_top + (h_pad % 2)
    return h_pad_top, h_pad_bottom, w_pad_left, w_pad_right
def validate(config, name, model_restored, val_loader, record_dict, loss_fn):
    Train = config['TRAINING']
    val_dir = Train['VAL'][f'{name}_DIR']
    gt_path = os.path.join(val_dir, 'gt')
    input_path = Train['VAL'][f'{name}_SAVE']
    mask_path = os.path.join(val_dir, 'mask')

    model_dir = os.path.join(Train['SAVE_DIR'], config['MODEL']['MODE'], 'models')

    model_restored.eval()
    print(f'==> Validation on {name} dataset=====================================================')

    for ii, data_val in enumerate(val_loader, 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        b, c, h, w = input_.size()

        # 预计算padding
        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = precompute_padding(h, w)

        # 使用反射填充可能比零填充更好
        input_padded = torch.nn.functional.pad(input_,(w_pad_left, w_pad_right, h_pad_top, h_pad_bottom))

        with torch.no_grad():
            restored = model_restored(input_padded)

            # 移除padding
            if h_pad_top + h_pad_bottom > 0:
                restored = restored[:, :, h_pad_top:-h_pad_bottom or None, :]
            if w_pad_left + w_pad_right > 0:
                restored = restored[:, :, :, w_pad_left:-w_pad_right or None]

        # 使用更高效的clamp和转换
        restored = torch.clamp(restored, 0, 1).mul(255).byte()  # 直接转为0-255的byte

        # 批量处理图像保存
        restored_np = restored.permute(0, 2, 3, 1).cpu().numpy()  # 移除不必要的detach()

        # 使用多线程保存图像
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = []
            for batch in range(b):
                img_bgr = cv2.cvtColor(restored_np[batch], cv2.COLOR_RGB2BGR)
                save_path = os.path.join(input_path, data_val[2][batch] + '.png')
                futures.append(executor.submit(cv2.imwrite, save_path, img_bgr))
            # 等待所有保存操作完成
            for future in futures:
                future.result()
    psnr_val_rgb, ssim_val_rgb, lpips_val_rgb, score_val_rgb, Gpsnr_val_rgb, Spsnr_val_rgb = calculate_metrics(
        gt_path, input_path, mask_path, loss_fn)
    assert not (math.fabs(psnr_val_rgb-10.6835)<1e-5), "nan or inf in PSNR calculation"
    # Save the best PSNR model of validation
    if psnr_val_rgb > record_dict['best_psnr']:
        record_dict['best_psnr'] = psnr_val_rgb
        record_dict['best_epoch_psnr'] = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_bestPSNR_{name}.pth"))
    print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
        epoch, psnr_val_rgb, record_dict['best_epoch_psnr'], record_dict['best_psnr']))

    # Save the best SSIM model of validation
    if ssim_val_rgb > record_dict['best_ssim']:
        record_dict['best_ssim'] = ssim_val_rgb
        record_dict['best_epoch_ssim'] = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_bestSSIM_{name}.pth"))
    print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
        epoch, ssim_val_rgb, record_dict['best_epoch_ssim'], record_dict['best_ssim']))

    # Save the best LPIPS model of validation
    if lpips_val_rgb < record_dict['best_lpips']:
        record_dict['best_lpips'] = lpips_val_rgb
        record_dict['best_epoch_lpips'] = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_bestLPIPS_{name}.pth"))
    print("[epoch %d LPIPS: %.4f --- best_epoch %d Best_LPIPS %.4f]" % (
        epoch, lpips_val_rgb, record_dict['best_epoch_lpips'], record_dict['best_lpips']))
    # Save the best score model of validation
    if score_val_rgb > record_dict['best_score']:
        record_dict['best_score'] = score_val_rgb
        record_dict['best_epoch_score'] = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_bestScore_{name}.pth"))
    print("[epoch %d Score: %.4f --- best_epoch %d Best_Score %.4f]" % (
        epoch, score_val_rgb, record_dict['best_epoch_score'], record_dict['best_score']))

    # Save the best Gpsnr model of validation
    if Gpsnr_val_rgb > record_dict['best_Gpsnr']:
        record_dict['best_Gpsnr'] = Gpsnr_val_rgb
        record_dict['best_epoch_Gpsnr'] = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_bestGpsnr_{name}.pth"))
    print("[epoch %d Gpsnr: %.4f --- best_epoch %d Best_Gpsnr %.4f]" % (
        epoch, Gpsnr_val_rgb, record_dict['best_epoch_Gpsnr'], record_dict['best_Gpsnr']))

    # Save the best Spsnr model of validation
    if Spsnr_val_rgb > record_dict['best_Spsnr']:
        record_dict['best_Spsnr'] = Spsnr_val_rgb
        record_dict['best_epoch_Spsnr'] = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_bestSpsnr_{name}.pth"))
    print("[epoch %d Spsnr: %.4f --- best_epoch %d Best_Spsnr %.4f]" % (
        epoch, Spsnr_val_rgb, record_dict['best_epoch_Spsnr'], record_dict['best_Spsnr']))
    """ 
    # Save evey epochs of model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
    """

    writer.add_scalar(f'val/PSNR_{name}', psnr_val_rgb, epoch)
    writer.add_scalar(f'val/SSIM_{name}', ssim_val_rgb, epoch)
    writer.add_scalar(f'val/LPIPS_{name}', lpips_val_rgb, epoch)
    writer.add_scalar(f'val/Score_{name}', score_val_rgb, epoch)
    writer.add_scalar(f'val/Gpsnr_{name}', Gpsnr_val_rgb, epoch)
    writer.add_scalar(f'val/Spsnr_{name}', Spsnr_val_rgb, epoch)


if __name__ == '__main__':


    # Start training!
    print('==> Training start: ')
    best_real_dict = {
        "best_psnr": 0,
        "best_ssim": 0,
        "best_lpips": 1000,
        "best_Gpsnr": 0,
        "best_Spsnr": 0,
        "best_score": 0,
        "best_epoch_psnr": 0,
        "best_epoch_ssim": 0,
        "best_epoch_lpips": 0,
        "best_epoch_score": 0,
        "best_epoch_Gpsnr": 0,
        "best_epoch_Spsnr": 0
    }
    best_syn_dict = {
        "best_psnr": 0,
        "best_ssim": 0,
        "best_lpips": 1000,
        "best_Gpsnr": 0,
        "best_Spsnr": 0,
        "best_score": 0,
        "best_epoch_psnr": 0,
        "best_epoch_ssim": 0,
        "best_epoch_lpips": 0,
        "best_epoch_score": 0,
        "best_epoch_Gpsnr": 0,
        "best_epoch_Spsnr": 0
    }

    config, writer = load_config()
    fabric = init_torch_config(config)
    model_restored, checkpoint, optimizer, scheduler, start_epoch = load_model(config, fabric)

    if checkpoint is not None:
        best_real_dict = checkpoint['best_real_dict']
        best_syn_dict = checkpoint['best_syn_dict']
        print("load indices from checkpoint succeed.")
        del checkpoint
    else:
        print('No checkpoint found, starting from scratch.')

    train_loader, real_val_loader, syn_val_loader = get_data_loaders(config, fabric)
    total_start_time = time.time()
    # gt_path = "./dataset/Flare7Kpp/test_data/real/gt"
    # gt_path = "./dataset/Flare7Kpp/test_data/real/gt"

    Train = config['TRAINING']
    OPT = config['TRAINOPTIM']
    model_dir = os.path.join(Train['SAVE_DIR'], config['MODEL']['MODE'], 'models')
    combined_loss1 = CombinedLoss(Train['LOSS']).cuda()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
        if 'LIMITED' in config and config['LIMITED']:
            assertLimited()
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_ssim_loss = 0
        epoch_c1_loss = 0
        epoch_vgg_loss = 0
        epoch_freq_loss = 0
        train_id = 1

        model_restored.train()
        for i, data in enumerate(train_loader, 0):
            # Forward propagation
            # for param in model_restored.parameters():
            #     param.grad = None
            optimizer.zero_grad()
            target = data[0].cuda()
            input_ = data[1].cuda()
            restored = model_restored(input_)

            loss = combined_loss1(restored, target)
            # Back propagation
            # loss.backward()
            fabric.backward(loss)
            optimizer.step()
            if i % 500 == 499:
                print(f'epoch {epoch}, iter {i + 1} finished.===================================================')
        ## Evaluation (Validation)

        if fabric.is_global_zero:

            if epoch % Train['VAL_AFTER_EVERY'] == 0:
                validate(config,'REAL',model_restored,  real_val_loader, best_real_dict,loss_fn_alex)
                validate(config,'SYN',model_restored,  syn_val_loader, best_syn_dict,loss_fn_alex)
            print("------------------------------------------------------------------")
            print(
                "Epoch: {}\tTime: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time, scheduler.get_lr()[0]))

            combined_loss1.print_cumulative_loss()
            combined_loss1.clear_cumulative_loss()


            print("------------------------------------------------------------------")
            # Save the last model
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_real_dict': best_real_dict,
                        'best_syn_dict': best_syn_dict,
                        }, os.path.join(model_dir, "model_latest.pth"))

            writer.add_scalar('train/loss', epoch_loss, epoch)
            writer.add_scalar('train/ssim_loss', epoch_ssim_loss, epoch)
            writer.add_scalar('train/c1_loss', epoch_c1_loss, epoch)
            writer.add_scalar('train/vgg_loss', epoch_vgg_loss, epoch)
            writer.add_scalar('train/freq_loss', epoch_freq_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

        scheduler.step()
    writer.close()

    total_finish_time = (time.time() - total_start_time)  # seconds
    print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
