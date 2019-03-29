import torch as t
import torch.nn as nn
from network import *
from torch import optim
import itertools
import os
import torchvision as tv



class DTNmodel(nn.Module):
	def __init__(self, opt):
		super(DTNmodel, self).__init__()
		t.backends.cudnn.benchmark = True

		self.n_iter = 0
		self.opt = opt
		# gpu配置
		self.gpu_ids = [0, 1, 2, 3]
		self.device = t.device('cuda:0') if opt.gpu else t.device('cpu')

		self.decoder = init_net(Decoder(), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
		self.encoder = init_net(Encoder(), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
		self.net_D = init_net(Discriminator_SA(batch_size=opt.batch_size), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

		self.optimizer_G = optim.Adam(itertools.chain(self.decoder.parameters(), self.encoder.parameters()), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
		self.optimizer_D = optim.Adam(itertools.chain(self.net_D.parameters()), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))

		# loss
		self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
		self.consist = nn.MSELoss()
		self.dit = nn.MSELoss()


	def set_input(self, data_S, data_T):
		self.real_S = data_S.to(self.device)
		self.real_T = data_T.to(self.device)


	def forward(self):
		self.latent_real_S = self.encoder(self.real_S)
		self.fake_T = self.decoder(self.latent_real_S)
		self.rec_T = self.decoder(self.encoder(self.real_T))
		self.latent_fake_T = self.encoder(self.fake_T)

	def backward_D(self):
		pred_real = self.net_D(self.real_T)
		pred_fake = self.net_D(self.fake_T.detach())
		pred_rec = self.net_D(self.rec_T.detach())

		loss = self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_rec, False)

		loss.backward()

	def backward_G(self):

		loss1 = self.criterionGAN(self.net_D(self.fake_T), True, is_generator=True) + self.criterionGAN(self.net_D(self.rec_T), True, is_generator=True)
		loss2 = self.opt.gamma * self.consist(self.latent_real_S, self.latent_fake_T) + self.opt.alpha * self.dit(self.real_T, self.rec_T)

		loss = loss1 + loss2

		loss.backward()


	def optimize_parameters(self):
		self.n_iter += 1
		self.forward()

		if self.n_iter % self.opt.n_critic == 0:
			self.optimizer_G.zero_grad()
			self.backward_G()
			self.optimizer_G.step()

		self.optimizer_D.zero_grad()
		self.backward_D()
		self.optimizer_D.step()

		



















