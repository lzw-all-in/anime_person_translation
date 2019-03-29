from DTN import DTNmodel
from data_loader import get_loader
import visdom
import tqdm
import numpy as np
import torch as t
import torchvision as tv
from argments import DTN_Options

if __name__ == '__main__':

	parser = DTN_Options()
	opt = parser.initialize()
	parser.print_args(opt)
	# 模型初始化
	model = DTNmodel(opt)
	# 数据集初始化
	dataset_A = get_loader(opt.domain_A_path, opt)
	dataset_B = get_loader(opt.domain_B_path, opt)
	# 用于在visdom中观察模型效果
	vis = visdom.Visdom(env=opt.env)
	fix_data_A = None
	fix_data_B = None

	# 开始迭代训练
	for epoch in range(opt.iteration):

		# define loss variable and clean
		loss_D_A = t.Tensor([0.0])
		loss_D_B = t.Tensor([0.0])
		
		for i, data in tqdm.tqdm(enumerate(zip(dataset_A, dataset_B))):
			data_A = data[0]
			data_B = data[1]
			# visualize real images
			if i == 0 and epoch == 0:
				fix_data_A = data_A.clone()[:36]
				fix_data_B = data_B.clone()[:36]
				vis.images(fix_data_A.numpy() * 0.5 + 0.5, nrow=6, win='1', opts={'title':"real anime images"})
				vis.images(fix_data_B.numpy() * 0.5 + 0.5, nrow=6, win='2', opts={'title':"real person images"})

			# 输入数据
			model.set_input(data_A, data_B)
			# forward & backward
			model.optimize_parameters()

			# visualize fake images
			if i % opt.vis_per_iter == 0:
				# G_A动漫到人，G_B人到动漫
				fake_anime = model.decoder(model.encoder(fix_data_B))
				rec_anime = model.decoder(model.encoder(fix_data_A))

				vis.images(fake_anime.detach().cpu().numpy() * 0.5 + 0.5, nrow=6, win='3', opts={'title':"fake anime images"})
				vis.images(rec_anime.detach().cpu().numpy() * 0.5 + 0.5, nrow=6, win='4', opts={'title':"rec anime images"})


		
	print('--------ending--------')



	