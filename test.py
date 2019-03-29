from cyclegan_model import CycleGANModel
from argments import Options
from data_loader import get_loader
import visdom
import tqdm
import numpy as np
import torch as t
import torchvision as tv

if __name__ == '__main__':

	# 初始化各项参数
	parser = Options()
	opt = parser.initialize()
	parser.print_args(opt)

	# 模型初始化
	model = CycleGANModel(opt)
	model.load_model('iter-200_sn_sn_sumiter200.pth')


	dataset_A = iter(get_loader(opt.domain_A_path, opt))
	dataset_B = iter(get_loader(opt.domain_B_path, opt))
	path = './test_image/'
	# how many image you wanna make
	num = 20
	for i in tqdm.tqdm(range(num)):
		data_A = next(dataset_A)
		data_B = next(dataset_B)

		tv.utils.save_image(model.netG_A(data_A).detach().cpu(), path + '{}__person_test.jpg'.format(i), nrow=8, padding=2, normalize=True, range=(-0.5, 0.5))
		tv.utils.save_image(model.netG_B(data_B).detach().cpu(), path + '{}__anime_test.jpg'.format(i), nrow=8, padding=2, normalize=True, range=(-0.5, 0.5))

		tv.utils.save_image(data_B, path + '{}__person_real.jpg'.format(i), nrow=8, padding=2, normalize=True, range=(-0.5, 0.5))
		tv.utils.save_image(data_A, path + '{}__anime_real.jpg'.format(i), nrow=8, padding=2, normalize=True, range=(-0.5, 0.5))

	del dataset_A, dataset_B
	print('------------------------done----------------------')