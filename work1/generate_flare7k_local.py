
from matplotlib import pyplot as plt
import os

from generator import Flare_Image_Loader

transform_base= {
	'img_size': 512
}


transform_flare= {
    "scale_min": 0.8,
    "scale_max": 1.5,
    "translate": 300,
    "shear": 20
}

length = 8192

import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


def save_image_batch(args):
    """保存一批图像的辅助函数，用于多线程处理"""
    c, indices = args
    for i in indices:
        dic = loader[i]  # 按需加载数据
        gt, input = dic['gt'], dic['lq']

        plt.imsave(f'./dataset/flare7k_local/input/c{c}/{i}.png',
                   input.numpy().transpose(1, 2, 0))
        plt.imsave(f'./dataset/flare7k_local/gt/c{c}/{i}.png',
                   gt.numpy().transpose(1, 2, 0))

        if i % 100 == 0:
            print(f'Processed {i} images for channel {c}.')


def batch_indices(total, batch_size):
    """生成批次索引的生成器"""
    for i in range(0, total, batch_size):
        yield list(range(i, min(i + batch_size, total)))


if __name__ == '__main__':
    # 初始化loader
    loader = Flare_Image_Loader('./dataset/Flickr24K', transform_base,
                                transform_flare, length=length)
    loader.load_scattering_flare('./dataset/Flare7Kpp/Flare7K',
                                 './dataset/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
    loader.load_light_source('./dataset/Flare7Kpp/Flare7K',
                             './dataset/Flare7Kpp/Flare7K/Scattering_Flare/Light_Source')
    loader.load_reflective_flare('./dataset/Flare7Kpp/Flare7K',
                                 './dataset/Flare7Kpp/Flare7K/Reflective_Flare')

    channels = [0, 1, 2, 3]
    batch_size = 32  # 根据内存情况调整批次大小

    # 创建必要的目录
    for c in channels:
        os.makedirs(f'./dataset/flare7k_local/input/c{c}', exist_ok=True)
        os.makedirs(f'./dataset/flare7k_local/gt/c{c}', exist_ok=True)

    # 使用线程池处理图像保存
    with ThreadPoolExecutor(max_workers=8) as executor:  # 根据CPU核心数调整
        for c in channels:
            # 分批处理图像
            for batch in batch_indices(length, batch_size):
                executor.submit(save_image_batch, (c, batch))

    print("All images processed and saved.")