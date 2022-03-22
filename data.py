import os
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from skimage.util import random_noise

class BSDS300_images(VisionDataset):
    basedir = "BSDS300"
    train_file = "iids_train.txt"
    test_file = "iids_test.txt"
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    archive_filename = "BSDS300-images.tgz"

    def __init__(self, root,noise_type, train=True, transform=None, download=False, parameter={'sigma':25,
                'threshold':0.05,'lowerValue':5, 'upperValue':250,
                'amount':0.05,'salt_vs_pepper':0.5}):
        super(BSDS300_images, self).__init__(root, transform=transform)
        args = {'sigma':25,
                'threshold':0.05,'lowerValue':5, 'upperValue':250,
                'amount':0.05,'salt_vs_pepper':0.5}

        diff = set(parameter.keys()) - set(args.keys())
        if diff:
           print("Invalid args:",tuple(diff))
           return

        args.update(parameter)  
        self.train = train
        
        #noise_type
        self.noise_type = noise_type
        #parameter for gaussian noise
        self.sigma = args['sigma']/255
        
        #parameter for Salt and Pepper noise 
        self.threshold = args['threshold']
        self.lowerValue = args['lowerValue'] / 255 # 255 would be too high
        self.upperValue = args['upperValue'] / 255 # 0 would be too low

        #parameter for Salt and Pepper noise
        self.amount = args['amount']
        self.salt_vs_pepper = args['salt_vs_pepper']
        
        self.root = root
        images_basefolder = os.path.join(root, self.basedir, "images")
        subfolder = "train" if self.train else "test"
        self.image_folder = os.path.join(images_basefolder, subfolder)
        id_file = self.train_file if self.train else self.test_file
        self.id_path = os.path.join(root, self.basedir, id_file)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        
        self.ids = np.loadtxt(self.id_path).astype('int')
        self.transform = transform
    
    def _check_exists(self):
        return os.path.exists(self.id_path) and os.path.exists(self.image_folder)
    
    def download(self):
        if self._check_exists():
            print("Files already downloaded")
            return
        download_and_extract_archive(self.url, download_root=self.root, filename=self.archive_filename)

    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, str(self.ids[idx])) + ".jpg"
        im = Image.open(img_name)
        if self.transform:
            im = self.transform(im)
        if self.noise_type == 'Gaussian':
            im_noisy = im + torch.randn(im.size()) * self.sigma
        elif self.noise_type =='SnP':
            random_matrix = torch.randn(im.size())  
            im_noisy=deepcopy(im)
            im_noisy[random_matrix>=(1-self.threshold)] = self.upperValue
            im_noisy[random_matrix<=self.threshold] = self.lowerValue

        elif self.noise_type =='s&p':
            im_noisy = random_noise(im, self.noise_type, amount=self.amount, salt_vs_pepper=self.salt_vs_pepper)
        elif self.noise_type =='speckle':
            im_noisy = random_noise(im, self.noise_type, var=self.sigma)
            im_noisy=im_noisy.astype('float32')
        elif self.noise_type == 'poisson':
            im_noisy = random_noise(im, self.noise_type)
            im_noisy=im_noisy.astype('float32')

        return im_noisy, im


class BerkeleyLoader(data.DataLoader):
    def __init__(self,noise_type, train=True,parameter={'sigma':25,'threshold':0.05,'lowerValue':5, 'upperValue':250,'amount':0.05,'salt_vs_pepper':0.5}, **kwargs):
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(90, pad_if_needed=True, padding_mode="reflect"),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()
        dataset = BSDS300_images("data",noise_type, train=train, transform=transform, download=True, parameter=parameter)
        super(BerkeleyLoader, self).__init__(dataset, **kwargs)



if __name__ == "__main__":
    train_data = BerkeleyLoader(25, train=True, batch_size=10)
    for im_noisy, im in train_data:
        print(im_noisy.size(), im.size())