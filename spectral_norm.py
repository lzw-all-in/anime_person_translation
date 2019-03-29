import torch as t
import torch.optim as optim
import torch.nn as nn
from torch.nn import Parameter

# spectralNorm效果确实比别的方法训练快也更好，但是太容易过拟合了，不知道是怎么回事

class SpectralNorm(nn.Module):

	def __init__(self, module, name='weight', power_iterations=1):
		super(SpectralNorm, self).__init__()
		self.module = module
		self.name = name
		self.power_iterations = power_iterations
		if not self._made_params():
			self._make_params()


	@staticmethod
	def l2normalize(v, eps=1e-12):
		return v / (v.norm() + eps)


	def _update_u_v(self):
		u = getattr(self.module, self.name + '_u')
		v = getattr(self.module, self.name + '_v')
		w = getattr(self.module, self.name + '_matrices')

		height = w.data.shape[0]
		for _ in range(self.power_iterations):
			# t.mv代表mat与vec相乘
			# 由于w是parameter类型的变量,使用data之后变为Tensor类型
			# 此处如果不是使用v.data = 的话，v会被覆盖为Tensor而不是parameter类型
			v.data = self.l2normalize(t.mv(w.view(height, -1).data.transpose(0, 1), u.data))
			u.data = self.l2normalize(t.mv(w.view(height, -1).data, v.data))
		# dot函数是对两个一维的tensor进行inner product并且是不满足broadast
		sigma = u.dot(w.view(height, -1).mv(v))
		# 这里重新赋值的变量其实就是论文里面的Wsn，谱归一化后得到的W
		setattr(self.module, self.name, w / sigma.expand_as(w))

	def _made_params(self):
		try:
			u = getattr(self.module, self.name + '_u')
			v = getattr(self.module, self.name + '_v')
			w = getattr(self.module, self.name + '_matrices')
			return True
		except AttributeError:
			return False

	def _make_params(self):
		w = getattr(self.module, self.name)
		# height = dout , width = din * w * h
		height = w.data.shape[0]
		width = w.view(height, -1).data.shape[1]
		# 使用type_as会让数据类型和cuda/cpu与该变量相同
		u = Parameter(t.randn(height), requires_grad=False)
		v = Parameter(t.randn(width), requires_grad=False)
		
		u.data = self.l2normalize(u.data)
		v.data = self.l2normalize(v.data)
		# 这里w是parameter类别的，但是data是Tensor类别, 默认是会有梯度的
		w_real = Parameter(w.data)
		# python的引用机制不一样, 即便删掉了这个，也只是删去了一个引用而已
		# w还在引用原来的变量，并且会随BP而更新变量值
		del self.module._parameters[self.name]
		# Adds a parameter to the module.
		self.module.register_parameter(self.name + '_u', u)
		self.module.register_parameter(self.name + '_v', v)
		self.module.register_parameter(self.name + '_matrices', w_real)
		
	def forward(self, *args):
		self._update_u_v()
		return self.module.forward(*args)		

