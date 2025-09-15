import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random

import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch

class RandomGammaCorrection(object):
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0.5, 1, 2]
            self.gamma = random.choice(gammas)
            return TF.adjust_gamma(image, self.gamma, gain=1)
        elif isinstance(self.gamma, tuple):
            gamma = random.uniform(*self.gamma)
            return TF.adjust_gamma(image, gamma, gain=1)
        elif self.gamma == 0:
            return image
        else:
            return TF.adjust_gamma(image, self.gamma, gain=1)


def remove_background(image):
    # the input of the image is PIL.Image form with [H,W,C]
    image = np.float32(np.array(image))
    _EPS = 1e-7
    rgb_max = np.max(image, (0, 1))
    rgb_min = np.min(image, (0, 1))
    image = (image - rgb_min) * rgb_max / (rgb_max - rgb_min + _EPS)
    image = torch.from_numpy(image)
    return image


def glod_from_folder(folder_list, index_list):
    ext = ['png', 'jpeg', 'jpg', 'bmp', 'tif']
    index_dict = {}
    for i, folder_name in enumerate(folder_list):
        data_list = []
        [data_list.extend(glob.glob(folder_name + '/*.' + e)) for e in ext]
        # data_list.sort()
        index_dict[index_list[i]] = data_list
    return index_dict


class Flare_Image_Loader(data.Dataset):
    def __init__(self, image_path, transform_base, transform_flare, mask_type=None,length=8192):
        self.ext = ['png', 'jpeg', 'jpg', 'bmp', 'tif']
        self.data_list = []
        [self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
        self.random_choices_gt = self.generate_random_indices(length, len(self.data_list))


        self.flare_dict = {}
        self.flare_name_list = []
        self.random_choices_flare = {}

        self.reflective_flag = False
        self.reflective_dict = {}
        self.reflective_name_list = []
        self.random_choices_reflective = {}

        self.light_flag = False
        self.light_dict = {}
        self.light_name_list = []
        self.length = length

        self.transform_r = transforms.CenterCrop((1008, 756))

        self.mask_type = mask_type  # It is a str which may be None,"luminance" or "color"
        self.img_size = transform_base['img_size']

        self.transform_output = transforms.Resize

        self.transform_base = transforms.Compose(
            [transforms.RandomCrop((self.img_size, self.img_size), pad_if_needed=True, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip()
             ])

        self.transform_flare = transforms.Compose([transforms.RandomAffine(degrees=(0, 360), scale=(
        transform_flare['scale_min'], transform_flare['scale_max']), translate=(
        transform_flare['translate'] / 1008, transform_flare['translate'] / 756), shear=(
        -transform_flare['shear'], transform_flare['shear'])),
                                                   transforms.CenterCrop((self.img_size, self.img_size)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomVerticalFlip()
                                                   ])
        self.data_ratio = []
        print("Base Image Loaded with examples:", len(self.data_list))

    def getI(self, index, is_real=False, with_reflective=False):
        return self.__getitem__(index, is_real, with_reflective)
    def __getitem__(self, index,is_real=False, with_reflective=False):
        # load base image
        img_path = self.data_list[index]
        # img_path = self.data_list[self.random_choices_gt[index]]
        base_img = Image.open(img_path).convert('RGB')

        gamma = np.random.uniform(1.8, 2.2)
        to_tensor = transforms.ToTensor()
        adjust_gamma = RandomGammaCorrection(gamma)
        adjust_gamma_reverse = RandomGammaCorrection(1 / gamma)
        color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)
        if self.transform_base is not None:
            base_img = to_tensor(base_img)
            base_img = adjust_gamma(base_img)
            base_img = self.transform_base(base_img)
        else:
            base_img = to_tensor(base_img)
            base_img = adjust_gamma(base_img)
        sigma_chi = 0.01 * np.random.chisquare(df=1)
        base_img = Normal(base_img, sigma_chi).sample()
        gain = np.random.uniform(0.5, 1.2)
        flare_DC_offset = np.random.uniform(-0.02, 0.02)
        base_img = gain * base_img
        base_img = torch.clamp(base_img, min=0, max=1)

        flare_list = self.flare_dict['r'] if is_real else self.flare_dict['7k']
        light_list = self.light_dict['r'] if is_real else self.light_dict['7k']
        reflective_list = self.reflective_dict['7k'] if with_reflective else None
        random_choices_flare = self.random_choices_flare['r'] if is_real else self.random_choices_flare['7k']
        random_choices_reflective = self.random_choices_reflective['7k'] if with_reflective else None
        # load flare and light source image
        light_img=None
        if self.light_flag:
            assert len(flare_list) == len(
                light_list), "Error, number of light source and flares dataset no match!"

            flare_path = flare_list[random_choices_flare[index]]
            light_path = light_list[random_choices_flare[index]]
            light_img = Image.open(light_path).convert('RGB')
            light_img = self.transform_r(light_img)
            light_img = to_tensor(light_img)
            light_img = adjust_gamma(light_img)
        else:
            flare_path = flare_list[random_choices_flare[index]]
        flare_img = Image.open(flare_path).convert('RGB')
        flare_img = self.transform_r(flare_img)
        reflective_img=None
        if self.reflective_flag and with_reflective:
            reflective_path = reflective_list[random_choices_reflective[index]]
            reflective_img = Image.open(reflective_path).convert('RGB')

        flare_img = to_tensor(flare_img)
        flare_img = adjust_gamma(flare_img)

        if self.reflective_flag and reflective_img is not None:
            reflective_img = self.transform_r(reflective_img)
            reflective_img = to_tensor(reflective_img)
            reflective_img = adjust_gamma(reflective_img)
            flare_img = torch.clamp(flare_img + reflective_img, min=0, max=1)

        flare_img = remove_background(flare_img)
        flare_merge=None
        if self.transform_flare is not None:
            if self.light_flag:
                flare_merge = torch.cat((flare_img, light_img), dim=0)
                flare_merge = self.transform_flare(flare_merge)
            else:
                flare_img = self.transform_flare(flare_img)

        # change color
        if self.light_flag:
            # flare_merge=color_jitter(flare_merge)
            flare_img, light_img = torch.split(flare_merge, 3, dim=0)
        else:
            flare_img = color_jitter(flare_img)

        # flare blur
        blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))
        flare_img = blur_transform(flare_img)
        # flare_img=flare_img+flare_DC_offset
        flare_img = torch.clamp(flare_img, min=0, max=1)

        # merge image
        merge_img = flare_img + base_img
        merge_img = torch.clamp(merge_img, min=0, max=1)
        if self.light_flag:
            base_img = base_img + light_img
            base_img = torch.clamp(base_img, min=0, max=1)
            flare_img = flare_img - light_img
            flare_img = torch.clamp(flare_img, min=0, max=1)

        if self.mask_type == None:
            return {'gt': adjust_gamma_reverse(base_img), 'flare': adjust_gamma_reverse(flare_img),
                    'lq': adjust_gamma_reverse(merge_img), 'gamma': gamma}
        elif self.mask_type == "luminance":
            # calculate mask (the mask is 3 channel)
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            luminance = 0.3 * flare_img[0] + 0.59 * flare_img[1] + 0.11 * flare_img[2]
            threshold_value = 0.99 ** gamma
            flare_mask = torch.where(luminance > threshold_value, one, zero)

        elif self.mask_type == "color":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            threshold_value = 0.99 ** gamma
            flare_mask = torch.where(merge_img > threshold_value, one, zero)
        elif self.mask_type == "flare":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            threshold_value = 0.7 ** gamma
            flare_mask = torch.where(flare_img > threshold_value, one, zero)
        elif self.mask_type == "light":
            # Depreciated: we dont need light mask anymore
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            luminance = 0.3 * light_img[0] + 0.59 * light_img[1] + 0.11 * light_img[2]
            threshold_value = 0.01
            flare_mask = torch.where(luminance > threshold_value, one, zero)
        return {'gt': adjust_gamma_reverse(base_img), 'flare': adjust_gamma_reverse(flare_img),
                'lq': adjust_gamma_reverse(merge_img), 'mask': flare_mask, 'gamma': gamma}

    def __len__(self):
        return self.length

    def load_scattering_flare(self, flare_name, flare_path):
        flare_list = []
        [flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
        flare_list = sorted(flare_list)
        self.flare_name_list.append(flare_name)
        self.flare_dict[flare_name] = flare_list
        self.random_choices_flare[flare_name] = self.generate_random_indices(self.length, len(flare_list))
        len_flare_list = len(self.flare_dict[flare_name])
        if len_flare_list == 0:
            print("ERROR: scattering flare images are not loaded properly")
        else:
            print("Scattering Flare Image:", flare_name, " is loaded successfully with examples", str(len_flare_list))
        print("Now we have", len(flare_list), 'scattering flare images')


    def load_light_source(self, light_name, light_path):
        # The number of the light source images should match the number of scattering flares
        light_list = []
        [light_list.extend(glob.glob(light_path + '/*.' + e)) for e in self.ext]
        light_list = sorted(light_list)
        self.flare_name_list.append(light_name)
        self.light_dict[light_name] = light_list
        len_light_list = len(self.light_dict[light_name])

        if len_light_list == 0:
            print("ERROR: Light Source images are not loaded properly")
        else:
            self.light_flag = True
            print("Light Source Image:", light_name, " is loaded successfully with examples", str(len_light_list))
        print("Now we have", len(light_list), 'light source images')

    def load_reflective_flare(self, reflective_name, reflective_path):
        if reflective_path is None:
            reflective_list = []
        else:
            reflective_list = []
            [reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
            reflective_list = sorted(reflective_list)
        self.reflective_name_list.append(reflective_name)
        self.reflective_dict[reflective_name] = reflective_list
        self.random_choices_reflective[reflective_name]=self.generate_random_indices(self.length, len(reflective_list))
        len_reflective_list = len(self.reflective_dict[reflective_name])
        if len_reflective_list == 0:
            print("ERROR: reflective flare images are not loaded properly")
        else:
            self.reflective_flag = True
            print("Reflective Flare Image:", reflective_name, " is loaded successfully with examples",
                  str(len_reflective_list))

    def generate_random_indices(self, req, total):
        if req <= 0 or total <= 0:
            return []
        if req <= total:
            return random.sample(range(total), req)
        else:
            result = []
            remaining = req

            while remaining > 0:
                sample_size = min(remaining, total)
                result.extend(random.sample(range(total), sample_size))
                remaining -= sample_size

            return result

    def shuffle_indices(self):
        random.shuffle(self.random_choices_gt)
        for k,v in self.random_choices_flare.items():
            random.shuffle(v)
        for k,v in self.random_choices_reflective.items():
            random.shuffle(v)





class Flare7kpp_Pair_Loader(Flare_Image_Loader):
    def __init__(self, opt):
        Flare_Image_Loader.__init__(self, opt['image_path'], opt['transform_base'], opt['transform_flare'],
                                    opt['mask_type'])
        scattering_dict = opt['scattering_dict']
        reflective_dict = opt['reflective_dict']
        light_dict = opt['light_dict']

        # defualt not use light mask if opt['use_light_mask'] is not declared
        if 'data_ratio' not in opt or len(opt['data_ratio']) == 0:
            self.data_ratio = [1] * len(scattering_dict)
        else:
            self.data_ratio = opt['data_ratio']

        if len(scattering_dict) != 0:
            for key in scattering_dict.keys():
                self.load_scattering_flare(key, scattering_dict[key])
        if len(reflective_dict) != 0:
            for key in reflective_dict.keys():
                self.load_reflective_flare(key, reflective_dict[key])
        if len(light_dict) != 0:
            for key in light_dict.keys():
                self.load_light_source(key, light_dict[key])

from matplotlib import pyplot as plt
if __name__ == '__main__':
    transform_base = {
        'img_size': 512
    }

    transform_flare = {
        "scale_min": 0.8,
        "scale_max": 1.5,
        "translate": 300,
        "shear": 20
    }

    flare_image_loader = Flare_Image_Loader('./dataset/Flickr24K', transform_base, transform_flare)
    # flare_image_loader.load_scattering_flare('./dataset/Flare7Kpp/Flare7K','./dataset/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
    # flare_image_loader.load_light_source('./dataset/Flare7Kpp/Flare7K','./dataset/Flare7Kpp/Flare7K/Scattering_Flare/Light_Source')
    flare_image_loader.load_scattering_flare('./dataset/Flare7Kpp/Flare7K',
                                             './dataset/Flare7Kpp/Flare-R/Compound_Flare')
    flare_image_loader.load_light_source('./dataset/Flare7Kpp/Flare7K', './dataset/Flare7Kpp/Flare-R/Light_Source')
    flare_image_loader.load_reflective_flare('./dataset/Flare7Kpp/Flare7K',
                                             './dataset/Flare7Kpp/Flare7K/Reflective_Flare')

    # flare_image_loader.load_lightsource('./dataset/Flare7Kpp/Flare7K','./dataset/Flare7Kpp/Flare7K/Scattering_Flare/Core')
    # img_index= random.randint(0,len(flare_image_loader)-1)
    img_index = 99
    # test_base_img,test_flare_img,test_merge_img,flare_mask=flare_image_loader[img_index]
    dic = flare_image_loader[img_index]
    test_base_img, test_merge_img = dic['gt'], dic['lq']
    img_index = 99
    # test_base_img,test_flare_img,test_merge_img,flare_mask=flare_image_loader[img_index]
    dic = flare_image_loader[img_index]
    test_base_img2, test_merge_img2 = dic['gt'], dic['lq']

    print(img_index)
    # plt.imshow(test_flare_img.permute(1,2,0))
    # plt.show()
    plt.imshow(test_merge_img.permute(1, 2, 0))
    plt.show()
    plt.imshow(test_base_img.permute(1, 2, 0))
    plt.show()
    plt.imshow(test_merge_img2.permute(1, 2, 0))
    plt.show()
    plt.imshow(test_base_img2.permute(1, 2, 0))
    plt.show()
    # print(flare_mask)
