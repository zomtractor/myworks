import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms

from data import Flare_Image_Loader


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class LocalDataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, patch_size, length=8192):
        super(LocalDataLoaderTrain, self).__init__()

        inp_files=os.listdir(os.path.join(rgb_dir, 'input', 'c0'))
        tar_files=os.listdir(os.path.join(rgb_dir, 'gt', 'c0'))

        self.inp_filenames = [os.path.join(rgb_dir, 'input','c0', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'gt','c0', x) for x in tar_files if is_image_file(x)]
        # self.sizex = len(self.tar_filenames)  # get the size of target
        self.sizex = len(self.inp_filenames) if length is None else length

        self.random_indices = random.sample(range(len(self.inp_filenames)), k=self.sizex)
        self.ps = patch_size

    def __len__(self):
        # return self.sizex
        return self.sizex

    def __getitem__(self, index):
        ps = self.ps
        strs = random.choice(['c0', 'c1', 'c2', 'c3'])
        # inp_path = self.inp_filenames[index_].replace('c0', strs )
        # tar_path = self.tar_filenames[index_].replace('c0', strs )
        idx = self.random_indices[index]
        inp_path = self.inp_filenames[idx].replace('c0', strs)
        tar_path = self.tar_filenames[idx].replace('c0', strs)
        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        return tar_img, inp_img, filename


class RealtimeDataLoaderTrain(Dataset):
    transform_base = {
        'img_size': 512
    }

    transform_flare = {
        "scale_min": 0.8,
        "scale_max": 1.5,
        "translate": 300,
        "shear": 20
    }
    mode="Flare7Kpp"

    def __init__(self, conf,patch_size):
        self.output_size=patch_size
        self.loader = Flare_Image_Loader(conf['BACKGROUND_DIR'],self.transform_base,self.transform_flare,length=conf['LENGTH'])
        self.loader.load_scattering_flare('7k',os.path.join(conf['FLARE7KPP_DIR'],"Flare7K","Scattering_Flare","Compound_Flare"))
        self.loader.load_light_source('7k',os.path.join(conf['FLARE7KPP_DIR'],"Flare7K","Scattering_Flare","Light_Source"))
        self.loader.load_reflective_flare('7k', os.path.join(conf['FLARE7KPP_DIR'], "Flare7K", "Reflective_Flare"))
        self.mode = conf['MODE']
        if self.mode=='Flare7Kpp':
            self.loader.load_scattering_flare('r', os.path.join(conf['FLARE7KPP_DIR'], "Flare-R", "Compound_Flare"))
            self.loader.load_light_source('r',os.path.join(conf['FLARE7KPP_DIR'],"Flare-R","Light_Source"))
        self.real_threshold = conf['REAL_PROPORTION']
        self.reflective_threshold = conf['REFLECTIVE_PROPORTION']
        self.resize = transforms.Resize(patch_size, interpolation=transforms.InterpolationMode.NEAREST)
    def __len__(self):
        # return self.sizex
        return len(self.loader)

    def __getitem__(self, index):
        is_real = random.random() < self.real_threshold
        with_reflective = random.random() < self.reflective_threshold
        result_dict = self.loader.getI(index, is_real=is_real, with_reflective=with_reflective)
        gt,lq,flare = result_dict['gt'],result_dict['lq'],result_dict['flare']
        gt=self.resize(gt)
        lq=self.resize(lq)
        flare=self.resize(flare)
        return gt,lq,flare

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None,length=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        # self.sizex = len(self.tar_filenames)  # get the size of target
        self.sizex = len(self.inp_filenames) if length is None else length

        self.random_indices = random.sample(range(len(self.inp_filenames)), k=self.sizex)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = self.random_indices[index]
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Validate on center crop
        # if self.ps is not None:
        #     inp_img = TF.center_crop(inp_img, (ps, ps))
        #     tar_img = TF.center_crop(tar_img, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        return tar_img, inp_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')

        inp = TF.to_tensor(inp)
        return inp, filename
