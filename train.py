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

	# 加载预训练模型
	# model.load_model('iter-180_sn_sn_sumiter200.pth')

	# 查看模型结构
	# model.print_architecture()

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
				fake_person = model.netG_A(fix_data_A)
				fake_anime = model.netG_B(fix_data_B)
				rec_person = model.netG_A(fake_anime)
				rec_anime = model.netG_B(fake_person)

				vis.images(fake_person.detach().cpu().numpy() * 0.5 + 0.5, nrow=6, win='3', opts={'title':"fake person images"})
				vis.images(fake_anime.detach().cpu().numpy() * 0.5 + 0.5, nrow=6, win='4', opts={'title':"fake anime images"})
				vis.images(rec_person.detach().cpu().numpy() * 0.5 + 0.5, nrow=6, win='5', opts={'title':"rec person images"})
				vis.images(rec_anime.detach().cpu().numpy() * 0.5 + 0.5, nrow=6, win='6', opts={'title':"rec anime images"})

			# accumulate loss value
			loss_D_A += model.loss_D_A
			loss_D_B += model.loss_D_B
		
		# visualize loss
		vis.line(X=t.Tensor([epoch]), Y=loss_D_A / (opt.dataset_size // opt.batch_size), win='Loss_win_1', name='loss_D_A', update='append', opts={'title':"Loss"})
		vis.line(X=t.Tensor([epoch]), Y=loss_D_B / (opt.dataset_size // opt.batch_size), win='Loss_win_1', name='loss_D_B', update='append', opts={'title':"Loss"})

		# save images
		if (epoch + 1) % opt.every_save_images == 0:
			model.save_images(fix_data_A, fix_data_B, epoch)

		# save model
		if (epoch + 1) % opt.ckpt_num == 0:
			model.save_model(epoch + 1)

		# update learning rate
		model.update_lr()
		
	print('--------ending--------')



	