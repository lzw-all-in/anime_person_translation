import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
	def __init__(self, channel):
		super(Self_Attention, self).__init__()
		self.channel = channel
		self.conv_f = nn.Conv2d(in_channels=channel, out_channels=channel // 8, kernel_size=1)
		self.conv_g = nn.Conv2d(in_channels=channel, out_channels=channel // 8, kernel_size=1)
		self.conv_h = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)

		# This allows the network to first rely on the cues in the local neighborhood 
		# – since this is easier – and then gradually learn to assign more weight to the non-local evidence.
		self.gamma = nn.Parameter(t.zeros(1), requires_grad=True)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		"""
			inputs:
				x : inputs feature maps(B, C, W, H)
			returns:
				out : self attention + x
				attention prob matrix: (B, N, N)  N = W * H
		"""
		m, c, width, height = x.size()

		f = self.conv_f(x).view(m, -1, width*height).permute(0, 2, 1)
		g = self.conv_g(x).view(m, -1, width*height)
		h = self.conv_h(x).view(m, -1, width*height)

		# beat = (m, w*h, w*h) 代表每个像素点与同一张图片上的不同像素点之间的关联度
		beta = self.softmax(t.bmm(f, g))

		# 此处beta需要调整一下
		sa_feats = (t.bmm(h, beta.permute(0, 2, 1))).view(m, c, width, height)

		return self.gamma * sa_feats + x



if __name__ == '__main__':
	data = t.randn([5, 16, 5, 6])
	sa = Self_Attention(16)
	print(sa(data).shape)


