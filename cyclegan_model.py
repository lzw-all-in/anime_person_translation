import torch as t
import torch.nn as nn
from network import *
from torch import optim
import itertools
import os
import torchvision as tv

class CycleGANModel(nn.Module):

	def __init__(self, opt):
		super(CycleGANModel, self).__init__()
		# 开始就启用可以加快速度，而且没有什么花销
		t.backends.cudnn.benchmark = True
		# 用来记录迭代轮次，好控制generator进行更新
		self.n_iter = 0
		self.opt = opt
		# gpu配置
		self.gpu_ids = [0, 1, 2, 3]
		self.device = t.device('cuda:0') if opt.gpu else t.device('cpu')

		# 实际效果,UNET更好一些并且显存占用也更少
		if opt.generator_mode == 'unet':
			self.netG_A = init_net(UnetGenerator(opt.input_nc, opt.output_nc), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
			self.netG_B = init_net(UnetGenerator(opt.input_nc, opt.output_nc), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)	
		elif opt.generator_mode == 'sn':		
			self.netG_A = init_net(Generator_SA(batch_size=opt.batch_size), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
			self.netG_B = init_net(Generator_SA(batch_size=opt.batch_size), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)	

		if opt.gan_mode == 'sn' or opt.gan_mode == 'sn+gp':
			self.netD_A = init_net(Discriminator_SA(batch_size=opt.batch_size), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
			self.netD_B = init_net(Discriminator_SA(batch_size=opt.batch_size), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
		elif opt.gan_mode == 'wgangp':
			self.netD_A = init_net(Discriminator(opt.input_nc, opt.ndf), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
			self.netD_B = init_net(Discriminator(opt.input_nc, opt.ndf), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

		self.fake_A_pool = ImagePool(opt.pool_size)
		self.fake_B_pool = ImagePool(opt.pool_size)
		self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
		self.criterionCycle = nn.L1Loss()
		self.criterionIdt = nn.L1Loss()

		self.optimizer_G = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
		self.optimizer_D = optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
		# 调整lr
		self.schedulers = [get_scheduler(self.optimizer_G, opt), get_scheduler(self.optimizer_D, opt)]
		self.optimizers = [self.optimizer_G, self.optimizer_D]

	def set_input(self, data_A, data_B):
		self.real_A = data_A.to(self.device)
		self.real_B = data_B.to(self.device)

	def forward(self):
		self.fake_B = self.netG_A(self.real_A)
		self.rec_A = self.netG_B(self.fake_B)
		self.fake_A = self.netG_B(self.real_B)
		self.rec_B = self.netG_A(self.fake_A)

	def backward_D(self, netD, real, fake):
		pred_real = netD(real)
		pred_fake = netD(fake)

		loss_real = self.criterionGAN(pred_real, True)
		loss_fake = self.criterionGAN(pred_fake, False)

		if self.opt.gan_mode == 'wgangp' or self.opt.gan_mode == 'sn+gp':
			gradient_penalty_loss = cal_gradient_penalty(netD, real, fake)
			loss = loss_real + loss_fake + gradient_penalty_loss
		elif self.opt.gan_mode == 'sn':
			loss = loss_real + loss_fake
		else: 
			loss = 0.5 * (loss_real + loss_fake)

		loss.backward()
		# 如果最后直接返回loss就会导致cal_gradient_penalty里面计算图一直没有被释放
		# 因为每次迭代后的函数调用结束后返回loss，并没有把loss给注销掉,导致之前保留的计算图
		# 一直保存在显存中,造成显存泄露
		# 所以此处如果要返回可以返回item或者使用detach
		return loss.detach().cpu()

	def backward_D_A(self):
		# 个人认为这里detach好一些,感觉ImagePool里面可以使用copy()方法
		# clone()的作用并没有表现出来,可能是由于copy的执行过于麻烦
		fake_A = self.fake_A_pool.query(self.fake_A)
		self.loss_D_A = self.backward_D(self.netD_A, self.real_A, fake_A)

	def backward_D_B(self):
		fake_B = self.fake_B_pool.query(self.fake_B)
		self.loss_D_B = self.backward_D(self.netD_B, self.real_B, fake_B)

	def backward_G(self):
		# GAN Loss
		GAN_loss_A = self.criterionGAN(self.netD_A(self.fake_A), True, is_generator=True)
		GAN_loss_B = self.criterionGAN(self.netD_B(self.fake_B), True, is_generator=True)

		# Cycle Loss
		# 下面一句话才是查看tensor是在哪个设备上
		# print(self.real_A.device, self.rec_A.device)
		Cyc_loss_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
		Cyc_loss_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
		# Identity Loss
		# 加入identity loss 会大幅增加训练时间，还要占用大量内存
		# 其作用是为了保留原图的色彩
		if self.opt.use_idt:
			Idt_loss_A = self.criterionIdt(self.real_A, self.netG_B(self.real_A)) * self.opt.lambda_A * self.opt.lambda_idt
			Idt_loss_B = self.criterionIdt(self.real_B, self.netG_A(self.real_B)) * self.opt.lambda_B * self.opt.lambda_idt
			
			loss = GAN_loss_A + GAN_loss_B + Cyc_loss_A + Cyc_loss_B + Idt_loss_A + Idt_loss_A

		# 去掉identity loss 
		else:
			loss = GAN_loss_A + GAN_loss_B + Cyc_loss_A + Cyc_loss_B

		loss.backward()

	def optimize_parameters(self):
		self.n_iter += 1
		self.forward()
		# generator
		if self.n_iter % self.opt.n_critic == 0:
			self.optimizer_G.zero_grad()
			self.backward_G()
			self.optimizer_G.step()

		# discriminator		
		self.optimizer_D.zero_grad()
		self.backward_D_A()
		self.backward_D_B()
		self.optimizer_D.step()

	def update_lr(self):
		# 使用step后scheduler内部的epoch+1
		for scheduler in self.schedulers:
			scheduler.step()
		# param_groups里面是一个列表，列表里面存放多个字典，每个字典有一系列参数
		lr = self.optimizers[0].param_groups[0]['lr']
		print('learning rate = %.7f' % lr)

	def print_architecture(self):
		print(self.netG_B)
		print(self.netD_A)
		pass

	# 模型、图片的存储与加载
	def save_model(self, epoch):
		if os.path.exists('./model_ckpt') is False:
			os.mkdir('./model_ckpt')

		t.save(self.state_dict(), './model_ckpt/iter-{}_{}_{}_sumiter{}.pth'.format(epoch, self.opt.gan_mode, self.opt.generator_mode, self.opt.iteration))

	def load_model(self, path):
		 self.load_state_dict(t.load('./model_ckpt/'+path))

	def save_images(self, fix_data_A, fix_data_B, epoch):

		path = './save_images/' + self.opt.gan_mode + '_' + str(self.opt.dataset_size) + '/'
		if os.path.exists(path) is False:
			os.mkdir(path)

		tv.utils.save_image(self.netG_A(fix_data_A).detach().cpu(), path + '{}_epoch_person_test.jpg'.format(epoch), nrow=6, padding=2, normalize=True, range=(-0.5, 0.5))
		tv.utils.save_image(self.netG_B(fix_data_B).detach().cpu(), path + '{}_epoch_anime_test.jpg'.format(epoch), nrow=6, padding=2, normalize=True, range=(-0.5, 0.5))








