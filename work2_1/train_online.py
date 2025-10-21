import math
import os
import random
import shutil
import time
import warnings
import datetime

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from lightning.fabric import Fabric
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
import utils
from data import get_training_data, get_validation_data
from model import CombinedLoss
from utils import network_parameters, MinIOHelper
from utils.mask_utils import calculate_metrics
from warmup_scheduler import GradualWarmupScheduler
import threading

from validate import validate


def assertLimited():
    start_time = datetime.time(7, 00)  # 7:40
    end_time = datetime.time(8, 15)  # 8:10
    assert not (start_time <= datetime.datetime.now().time() <= end_time), "当前时间位于禁止时间段 7:40~8:10 内"
    print("assert passed")


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
    # torch.set_anomaly_enabled(True)
    # fabric = Fabric(accelerator="cuda", devices=2, strategy="ddp_find_unused_parameters_true")
    fabric = Fabric(accelerator="cuda", devices=config['TRAINOPTIM']['DEVICES'])
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
        try:
            path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
            checkpoint = utils.load_checkpoint(model_restored, path_chk_rest)
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
        except :
            print('checkpoint load failed, start from scratch.')

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

def minio_sync(minio_helper,model_dir,update_real_list,update_syn_list):
    minio_helper.upload_file(os.path.join(model_dir, "model_latest.pth"))
    for best_type in update_real_list:
        minio_helper.copy_file_local_remote(
            os.path.join(model_dir, "model_latest.pth"),
            os.path.join(model_dir, f"model_best_{best_type}_REAL.pth"))
    for best_type in update_syn_list:
        minio_helper.copy_file_local_remote(
            os.path.join(model_dir, "model_latest.pth"),
            os.path.join(model_dir, f"model_best_{best_type}_REAL.pth"))


if __name__ == '__main__':

    minio_helper = MinIOHelper(
            endpoint='objectstorageapi.ap-northeast-1.clawcloudrun.com',
            access_key='16bqw05c',
            secret_key='h8zdm5kg6k9kg26z',
            bucket_name="16bqw05c-mywork",
            secure=True)
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

    train_loader, real_val_loader, syn_val_loader = get_data_loaders(config, fabric)
    total_start_time = time.time()
    # gt_path = "./dataset/Flare7Kpp/test_data/real/gt"
    # gt_path = "./dataset/Flare7Kpp/test_data/real/gt"

    Train = config['TRAINING']
    OPT = config['TRAINOPTIM']
    model_dir = os.path.join(Train['SAVE_DIR'], config['MODEL']['MODE'], 'models')
    combined_gt_loss1 = CombinedLoss(Train['LOSS']).cuda()
    combined_flare_loss1 = CombinedLoss(Train['LOSS']).cuda()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
        if 'LIMITED' in config and config['LIMITED']:
            assertLimited()
        epoch_start_time = time.time()

        model_restored.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # Forward propagation
            # for param in model_restored.parameters():
            #     param.grad = None
            optimizer.zero_grad()
            target = data[0].cuda()
            input_ = data[1].cuda()
            flare = data[2].cuda()
            restored, flarepred = model_restored(input_)

            loss1_gt = combined_gt_loss1(restored, target)
            loss1_flare = combined_flare_loss1(flarepred, flare)
            loss = loss1_gt + 0.1 * loss1_flare
            # Back propagation
            # loss.backward()
            fabric.backward(loss)
            optimizer.step()
            if i % 500 == 499:
                print(f'epoch {epoch}, iter {i + 1} finished.===================================================')
        ## Evaluation (Validation)

        if fabric.is_global_zero:
            model_restored.eval()
            update_real_list = validate(epoch, config, 'REAL', lambda x:model_restored(x)[0], real_val_loader, best_real_dict, loss_fn_alex,writer)
            update_syn_list = validate(epoch,config, 'SYN', lambda x:model_restored(x)[0], syn_val_loader, best_syn_dict, loss_fn_alex,writer)
            print("------------------------------------------------------------------")
            print(
                "Epoch: {}\tTime: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                      scheduler.get_lr()[0]))
            combined_gt_loss1.print_cumulative_loss('gt')
            combined_gt_loss1.clear_cumulative_loss()
            combined_flare_loss1.print_cumulative_loss('flare')
            combined_flare_loss1.clear_cumulative_loss()

            print("------------------------------------------------------------------")
            # Save the last model
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_real_dict': best_real_dict,
                        'best_syn_dict': best_syn_dict,
                        }, os.path.join(model_dir, "model_latest.pth"))
            threading.Thread(target=lambda:minio_sync(minio_helper,model_dir,update_real_list,update_syn_list)).start()

        scheduler.step()
    writer.close()

    total_finish_time = (time.time() - total_start_time)  # seconds
    print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
