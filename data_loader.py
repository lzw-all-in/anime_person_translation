import os
from PIL import  Image
import numpy as np
from torchvision import transforms as T
from torch.utils import data


class ImageFolder(data.Dataset):
	def __init__(self, root, transforms=None, dataset_size=10000):
		imgs_path = os.listdir(root)[:dataset_size]
		self.imgs = [os.path.join(root, img) for img in imgs_path]
		self.transforms = transforms

	def __getitem__(self, index):
		img_path = self.imgs[index]
		label = None
		image = Image.open(img_path)
		if self.transforms:
			image = self.transforms(image)
		if label == None:
			return image
		else:
			return image, label

	def __len__(self):
		return len(self.imgs)

def get_loader(root, opt):

	transforms = T.Compose([
				T.Resize(64),
				T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1], 并且将channel换到第一维度
    			T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
			])

	dataset = ImageFolder(root=root, transforms=transforms, dataset_size=opt.dataset_size)

	return data.DataLoader(dataset, 
						batch_size=opt.batch_size, 
						shuffle=True, 
						num_workers=opt.num_workers, 
						drop_last=True)
