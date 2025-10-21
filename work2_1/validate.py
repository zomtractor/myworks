import os
import math
import cv2
import torch

from utils.mask_utils import calculate_metrics

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('result.log.txt'))

def validate(epoch, config, ds_type, fetch_input_fn, val_loader, record_dict, lpips_fn, writer):
    res=[]
    Train = config['TRAINING']
    val_dir = Train['VAL'][f'{ds_type}_DIR']
    gt_path = os.path.join(val_dir, 'gt')
    input_path = Train['VAL'][f'{ds_type}_SAVE']
    mask_path = os.path.join(val_dir, 'mask')
    print(f'==> Validation on {ds_type} dataset=====================================================')
    logger.info(f'==> Validation on {ds_type} dataset=====================================================')

    # 处理验证数据并保存图像
    for ii, data_val in enumerate(val_loader, 0):
        input_ = data_val[1].cuda()
        b, c, h, w = input_.size()
        with torch.no_grad():
            restored = fetch_input_fn(input_)
        restored = torch.clamp(restored, 0, 1).mul(255).byte()
        restored_np = restored.permute(0, 2, 3, 1).cpu().numpy()

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

    # 计算指标
    psnr_val_rgb, ssim_val_rgb, lpips_val_rgb, score_val_rgb, Gpsnr_val_rgb, Spsnr_val_rgb = calculate_metrics(
        gt_path, input_path, mask_path, lpips_fn)
    assert not (math.fabs(psnr_val_rgb - 10.6835) < 1e-5), "nan or inf in PSNR calculation"

    # 定义指标处理逻辑
    metrics = [
        ('psnr', psnr_val_rgb, True, 'PSNR'),
        ('ssim', ssim_val_rgb, True, 'SSIM'),
        ('lpips', lpips_val_rgb, False, 'LPIPS'),  # LPIPS是越低越好
        ('score', score_val_rgb, True, 'Score'),
        ('Gpsnr', Gpsnr_val_rgb, True, 'Gpsnr'),
        ('Spsnr', Spsnr_val_rgb, True, 'Spsnr')
    ]

    # 统一处理所有指标
    for metric_name, current_value, higher_is_better, display_name in metrics:
        best_metric_key = f'best_{metric_name}'
        best_epoch_key = f'best_epoch_{metric_name}'

        # 判断是否需要更新最佳值
        should_update = (current_value > record_dict[best_metric_key]) if higher_is_better else (
                    current_value < record_dict[best_metric_key])

        if should_update:
            record_dict[best_metric_key] = current_value
            record_dict[best_epoch_key] = epoch
            res.append(display_name)

        # 打印结果
        print(
            f"[epoch {epoch} {display_name}: {current_value:.4f} --- best_epoch {record_dict[best_epoch_key]} Best_{display_name} {record_dict[best_metric_key]:.4f}]")

        logger.info(
            f"[epoch {epoch} {display_name}: {current_value:.4f} --- best_epoch {record_dict[best_epoch_key]} Best_{display_name} {record_dict[best_metric_key]:.4f}]")

        # 记录到tensorboard
        writer.add_scalar(f'val/{display_name}_{ds_type}', current_value, epoch)

    return res