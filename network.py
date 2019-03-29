import torch as t
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import functools
import random
import numpy as np
import time
from spectral_norm import SpectralNorm
from self_attention import Self_Attention

def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'linear':
		# lambda的结果作为乘法因子与学习率相乘
		# 100轮迭代之前不下降，之后开始线性递减
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, (epoch - opt.niter) / float(opt.niter_decay + 1))
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	else:
		NotImplementedError('learning rate policy %s is not implemented' % (opt.lr_policy))
	
	return scheduler		

# 使用ImagePool为了减少震荡
class ImagePool():
	def __init__(self, pool_size):
		self.pool_size = pool_size
		if pool_size > 0:
			self.num_imgs = 0
			self.imgs = []

	def query(self, images):
		# 参数images是最近生成的图片
		# 0.5的概率返回输入的图片,0.5的概率返回buffer内的图片并插入当前图片
		if self.pool_size == 0:
			return images
		return_images = []
		for image in images:
			image = t.unsqueeze(image.data, 0) # 1, c, h, w
			# 如果pool没有装满那么返回当前图片即可，并将当前图片装入buffer
			if self.num_imgs < self.pool_size:
				self.imgs.append(image)
				return_images.append(image)
				self.num_imgs += 1
			else:
				p = random.uniform(0, 1)
				if p > 0.5:
					# 闭区间
					random_id = random.randint(0, self.pool_size-1)
					# 如果是使用copy()是不行的，该方法只是复制了它的值而已
					# 这样得到的image是没有computation graph的,这样bp就无法更新generator
					# 但是由于是更新discriminator的时候才使用这个Pool那么之前的计算图并没有用
					# 此处使用clone其实是因为copy方法使用起来过于麻烦，需要两个tensor，shape能够进行广播
					pre_image = self.imgs[random_id].clone()
					self.imgs[random_id] = image
					return_images.append(pre_image)
				else:
					return_images.append(image)
		return_images = t.cat(return_images, 0)
		return return_images.detach()



class GANLoss(nn.Module):
	"""docstring for GANLoss"""
	def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		# 这里为什么要存放在register_buffer中而不是直接赋值存放在变量中就是因为：
		# 当GAN_Loss放在cuda上的时候register_buffer也自动转换到cuda中了
		# 而self的一个变量却还是在cpu之中,如果使用下面的方式是会报错的
		self.register_buffer('real_label', t.tensor(target_real_label))
		self.register_buffer('fake_label', t.tensor(target_fake_label))
		# self.real_label = t.tensor(target_real_label)
		# self.fake_label = t.tensor(target_fake_label)
		self.gan_mode = gan_mode
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss()
		elif gan_mode == 'wgangp' or gan_mode == 'sn' or gan_mode == 'sn+gp':
			self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % (gan_mode))

	def get_target_tensor(self, prediction, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction)

	def forward(self, prediction, target_is_real, is_generator=False):
		if self.gan_mode in ['lsgan', 'vanilla']:
			target_tensor = self.get_target_tensor(prediction, target_is_real)
			loss = self.loss(prediction, target_tensor)
										# 谱归一化加wgan-gp, 或者hinge loss模式下的generator
		elif self.gan_mode == 'wgangp' or self.gan_mode == 'sn+gp' or is_generator:
			if target_is_real:
				loss = -prediction.mean()
			else:
				loss = prediction.mean()
		# 谱归一化使用hinge loss		
		elif self.gan_mode == 'sn':
			if target_is_real:		
				# loss = -prediction.mean()
				loss = (nn.ReLU(inplace=True)(1 - prediction)).mean()
			else:
				# loss = prediction.mean()
				loss = (nn.ReLU(inplace=True)(1 + prediction)).mean()
		return loss

def cal_gradient_penalty(netD, real_data, fake_data, device='cuda:0', type='mixed', constant=1.0, lambda_gp=10.0):
	if lambda_gp > 0.0:
		if type == 'real':
			interpolate_samples = real_data
		elif type == 'fake':
			interpolate_samples = fake_data
		elif type == 'mixed':
			alpha = t.rand(real_data.shape[0], 1, 1, 1).to(device)
			interpolate_samples = alpha * real_data + (1 - alpha) * fake_data
		else:
			raise NotImplementedError('this type is not implemented')

		interpolate_samples.requires_grad_(True)
		outputs = netD(interpolate_samples)

		gradients = t.autograd.grad(
	        outputs=outputs,
	        inputs=interpolate_samples,
	        grad_outputs=t.ones(outputs.size(), requires_grad=False).to(device),
	        create_graph=True,
	        retain_graph=True,
	        only_inputs=True,
	    )[0]
		# 梯度的shape与输入一致
		gradients = gradients.view(real_data.size(0), -1)
		gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp

		return gradient_penalty
	else:
		return 0.0

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(t.cuda.is_available())
		# 虽然要进行并行化，但是下面这个语句至关重要
		net.to(gpu_ids[0])
		if t.cuda.device_count() > 1:
			# 注：对于多GPU运行只需要下面一句代码即可
			net = t.nn.DataParallel(net, gpu_ids) 

	init_weights(net, init_type, init_gain)
	return net


def init_weights(net, init_type='normal', init_gain=0.02):
	# init_gain是缩放因子,控制范围
	def init_func(submodel):
		name = submodel.__class__.__name__
		# 字符串find方法找到第一个匹配的index再返回回来，如果没有找到则返回-1
		if hasattr(submodel, 'weight') and (name.find('Conv') != -1 or name.find('Linear') != -1):	
			if init_type == 'normal':
				init.normal_(submodel.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(submodel.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(submodel.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(submodel.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method %s is not implemented' % init_type)
			if hasattr(submodel, 'bias') and submodel.bias is not None:
				init.constant_(submodel.bias.data, 0.0)

		elif name.find('BatchNorm2d') != -1:
			# 注意batchnorm的 Weigh并不是一个矩阵
			init.normal_(submodel.weight.data, 0.0, init_gain)
			init.constant_(submodel.bias.data, 0.0)

	print('initialize network with %s' % (net.__class__.__name__))
	# apply函数是对其model和其submodel进行递归调用
	net.apply(init_func)


class ResnetBlock(nn.Module):
	"""docstring for ResnetBlock"""
	def __init__(self, c_dim, padding_type, norm_layer, use_dropout, use_bias=False):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(c_dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, c_dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding %s is not implemented' % (padding_type))

		conv_block += [nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(c_dim), nn.ReLU(True)]

		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding %s is not implemented' % (padding_type))

		conv_block += [nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(c_dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out

class UnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs=7, ngf=64, norm_layer=nn.InstanceNorm2d):
		super(UnetGenerator, self).__init__()
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True, norm_layer=norm_layer)
		for i in range(num_downs - 5):
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
		
	def forward(self, inputs):
		return self.model(inputs)


class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc, input_nc=None,
				submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if input_nc == None:
			input_nc = outer_nc
		# 下采样
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
							stride=2, padding=1, bias=False)
		downrelu = nn.LeakyReLU(0.2, inplace=True)
		downnorm = norm_layer(inner_nc)
		#上采样
		uprelu = nn.ReLU(inplace=True)
		upnorm = norm_layer(outer_nc)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=True)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=False)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=False)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]
			model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			# 对通道数进行concat	
			return t.cat([x, self.model(x)], 1)


class Generator_SA(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, norm_layer=nn.InstanceNorm2d):
        super(Generator_SA, self).__init__()
        # 下采样
        self.up_sample= nn.Sequential(
        			nn.Conv2d(3, conv_dim, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim),
        			nn.ReLU(inplace=False),
        			# Self_Attention(conv_dim),
        			nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim*2),
        			nn.ReLU(inplace=False),
        			# Self_Attention(conv_dim*2),
        			nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim*4),
        			nn.ReLU(inplace=False),
        			nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim*8),
        			nn.ReLU(inplace=False),
        			nn.Conv2d(conv_dim*8, 100, kernel_size=4, padding=0, stride=1),
        			nn.ReLU(inplace=True)
        			)
        # 上采样
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(norm_layer(conv_dim * mult))
        layer1.append(nn.ReLU(inplace=False))

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(norm_layer(int(curr_dim / 2)))
        layer2.append(nn.ReLU(inplace=False))

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(norm_layer(int(curr_dim / 2)))
        layer3.append(nn.ReLU(inplace=False))

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(norm_layer(int(curr_dim / 2)))
            layer4.append(nn.ReLU(inplace=False))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.up_attn1 = Self_Attention(128)
        self.up_attn2 = Self_Attention(64)

    def forward(self, x):
        z = self.up_sample(x)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out = self.up_attn1(out)
        out = self.l4(out)
        out = self.up_attn2(out)
        out = self.last(out)

        return out

# ------------------------------- Discriminator -------------------------------------

# 这里的InstanceNorm2d也出现了bug，如果采取它的话，bp两次就算设置retain_graph也会报错
# 并且错误居然是没有让retain_graph=True，于是改用了BN就没有错误了
# 目前而言得到的结论是InstanceNorm与inplace=True不能共存
class Discriminator(nn.Module):
	# 所谓的PatchGAN其实其本质就是一个ConvNet而已，看最后输出的一个像素来源于最开始图像的多大的区域
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_bias=False):
		super(Discriminator, self).__init__()

		ks = 4
		padding = 1
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=ks, stride=2, padding=padding, bias=True), nn.LeakyReLU(0.2, inplace=True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2**n, 8)
			sequence += [
				nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks, stride=2, padding=padding, bias=use_bias),
				norm_layer(ndf*nf_mult),
				nn.LeakyReLU(0.2, inplace=False)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2**n_layers, 8)
		sequence += [
				nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=ks, stride=1, padding=padding, bias=use_bias),
				norm_layer(ndf*nf_mult),
				nn.LeakyReLU(0.2, inplace=False)
		]

		sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=ks, stride=1, padding=padding, bias=True)]
		self.model = nn.Sequential(*sequence)

	def forward(self, inputs):
		return self.model(inputs)


class Discriminator_SA(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator_SA, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attention(256)
        self.attn2 = Self_Attention(512)

     # 实验得出对于判别器而言attention加在后面会比前面好
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)

        return out.squeeze()

class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, conv_dim=64, dim_z=100, norm_layer=nn.InstanceNorm2d):
		super(Encoder, self).__init__()
		self.down_sample= nn.Sequential(
        			nn.Conv2d(3, conv_dim, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim),
        			nn.ReLU(inplace=False),
        			# Self_Attention(conv_dim),
        			nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim*2),
        			nn.ReLU(inplace=False),
        			# Self_Attention(conv_dim*2),
        			nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim*4),
        			nn.ReLU(inplace=False),
        			nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, padding=1, stride=2),
        			norm_layer(conv_dim*8),
        			nn.ReLU(inplace=False),
        			nn.Conv2d(conv_dim*8, dim_z, kernel_size=4, padding=0, stride=1),
        			nn.ReLU(inplace=True)
        			)

	def forward(self, x):
		return self.down_sample(x)


class Decoder(nn.Module):
    """Generator."""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64, norm_layer=nn.InstanceNorm2d):
        super(Decoder, self).__init__()
        # 上采样
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(norm_layer(conv_dim * mult))
        layer1.append(nn.ReLU(inplace=False))

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(norm_layer(int(curr_dim / 2)))
        layer2.append(nn.ReLU(inplace=False))

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(norm_layer(int(curr_dim / 2)))
        layer3.append(nn.ReLU(inplace=False))

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(norm_layer(int(curr_dim / 2)))
            layer4.append(nn.ReLU(inplace=False))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.up_attn1 = Self_Attention(128)
        self.up_attn2 = Self_Attention(64)

    def forward(self, z):
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out = self.up_attn1(out)
        out = self.l4(out)
        out = self.up_attn2(out)
        out = self.last(out)

        return out

if __name__ == '__main__':
	dis = Discriminator_SA()
	data = t.randn([64, 3, 64, 64])
	print(dis(data).shape)