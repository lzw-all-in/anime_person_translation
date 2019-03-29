import argparse

class  Options():
	"""docstring for  Options"""
	def __init__(self):
		pass

	def initialize(self):
		parser = argparse.ArgumentParser()

		# 调整学习率参数
		parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
		parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
		parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy')
        
        # Optimizer
		parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
		parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')		
		parser.add_argument('--n_critic', type=int, default=1, help='momentum term of adam')		
		parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for adam')
		parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for adam')
		parser.add_argument('--generator_mode', type=str, default='sn', help='choose the architechture of the generator')

		# Loss parameter
		parser.add_argument('--lambda_A', type=float, default=250.0, help='weight for cycle loss (A -> B -> A)')
		parser.add_argument('--lambda_B', type=float, default=250.0, help='weight for cycle loss (B -> A -> B)')
		parser.add_argument('--lambda_idt', type=float, default=0.05, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
		parser.add_argument('--use_idt', help='use identity loss or not', action='store_true', default=False)		
		
		# GPU
		parser.add_argument('--gpu', help='use gpu or not', action='store_true', default=True)
		
		# model parameters
		parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
		parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
		parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
		parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
		parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
		parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
		parser.add_argument('--gan_mode', type=str, default='sn', help='the type of GAN objective. [vanilla| lsgan | wgangp | sn | sn+gp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
		parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
		
		# dataloader parameters
		parser.add_argument('--num_workers', type=int, default=4, help='# of processor in data loading')
		parser.add_argument('--batch_size', type=int, default=64, help='# of batch size')
		parser.add_argument('--domain_A_path', type=str, default='../GAN_data/data/animation/', help='the image path')
		parser.add_argument('--domain_B_path', type=str, default='../GAN_data/data/person/', help='the image path| ../GAN_data/data/person/| ../GAN_data/data/asia_woman/')
		parser.add_argument('--every_save_images', type=int, default=10, help='# of epoch to save image')
		parser.add_argument('--save_person_path', type=str, default='', help='the path of save images')
		parser.add_argument('--dataset_size', type=int, default=50000, help='the scale of dataset')
		
		# original parameters
		parser.add_argument('--iteration', type=int, default=200, help='# of epochs')
		parser.add_argument('--ckpt_num', type=int, default=20, help='# of epochs to save the model')
		
		# visualize parameters
		parser.add_argument('--vis_per_iter', type=int, default=20, help='# of iteration that update the visdom')
		parser.add_argument('--env', type=str, default='GAN_CyC', help='enviroment of the visualize images')

		self.parser = parser
		return parser.parse_args()

	def print_args(self, opt):
		message = ''
		message += '----------------- Options ---------------\n'
		# vars返回由输入对象的属性与属性值组成的字典
		for k, v in sorted(vars(opt).items()):
		    comment = ''
		    default = self.parser.get_default(k)
		    if v != default:
		        comment = '\t[default: %s]' % str(default)
		        # {:>25}是指字符串至少要有25的lenth,不足处左边补空格
		        # {:<30}字符串至少30的lenth，不足处右边补空格
		    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------\n'
		print(message)



class  DTN_Options():
	def __init__(self):
		pass

	def initialize(self):
		parser = argparse.ArgumentParser()

		# 调整学习率参数
		parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy')
        
        # Optimizer
		parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
		parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')		
		parser.add_argument('--n_critic', type=int, default=1, help='momentum term of adam')		
		parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for adam')
		parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for adam')

		# loss
		parser.add_argument('--gamma', type=float, default=10.0, help='gamma')
		parser.add_argument('--alpha', type=float, default=300.0, help='alpha')

		# GPU
		parser.add_argument('--gpu', help='use gpu or not', action='store_true', default=True)
		
		# model parameters
		parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
		parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
		parser.add_argument('--gan_mode', type=str, default='sn', help='the type of GAN objective. [vanilla| lsgan | wgangp | sn | sn+gp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
		
		# dataloader parameters
		parser.add_argument('--num_workers', type=int, default=4, help='# of processor in data loading')
		parser.add_argument('--batch_size', type=int, default=64, help='# of batch size')
		parser.add_argument('--domain_A_path', type=str, default='../GAN_data/data/animation/', help='the image path')
		parser.add_argument('--domain_B_path', type=str, default='../GAN_data/data/person/', help='the image path| ../GAN_data/data/person/| ../GAN_data/data/asia_woman/')
		parser.add_argument('--every_save_images', type=int, default=10, help='# of epoch to save image')
		parser.add_argument('--dataset_size', type=int, default=50000, help='the scale of dataset')
		
		# original parameters
		parser.add_argument('--iteration', type=int, default=200, help='# of epochs')
		parser.add_argument('--ckpt_num', type=int, default=20, help='# of epochs to save the model')
		
		# visualize parameters
		parser.add_argument('--vis_per_iter', type=int, default=20, help='# of iteration that update the visdom')
		parser.add_argument('--env', type=str, default='GAN_DTN', help='enviroment of the visualize images')

		self.parser = parser
		return parser.parse_args()

	def print_args(self, opt):
		message = ''
		message += '----------------- Options ---------------\n'
		# vars返回由输入对象的属性与属性值组成的字典
		for k, v in sorted(vars(opt).items()):
		    comment = ''
		    default = self.parser.get_default(k)
		    if v != default:
		        comment = '\t[default: %s]' % str(default)
		        # {:>25}是指字符串至少要有25的lenth,不足处左边补空格
		        # {:<30}字符串至少30的lenth，不足处右边补空格
		    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------\n'
		print(message)

if __name__ == '__main__':
	pass
