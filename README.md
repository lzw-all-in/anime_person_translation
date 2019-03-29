# 人脸与动漫脸互转

### 基本说明

1. 代码中主要尝试了2种框架 [cycleGAN](<https://arxiv.org/abs/1703.10593>) 和 [DTN](<https://arxiv.org/abs/1611.02200>) 但是DTN的效果太差甚至跑不起来
2. 尝试了各种优化方式比如：[WGAN](<https://arxiv.org/abs/1701.07875>) [WGAN-GP](<https://arxiv.org/abs/1704.00028v3>) [SN](<https://arxiv.org/abs/1802.05957>) 最后得出结论虽然SN好与前两者，但是存在很严重的mode collapse的问题，就是无论输入什么样的人脸输出的动漫脸都是同一张
3. 最后在cycleGAN的框架之上采用了[self-attention GAN](<https://arxiv.org/abs/1805.08318>) 才最终解决问题，也就是我的最终版本
4. 数据集 人脸数据是 [CelebA](https://pan.baidu.com/s/17h5f2eOGSTr_iabmSOA2Iw) ，动漫脸数据是 [知乎上有人已经爬取下来的](<https://pan.baidu.com/s/1eSifHcA>) 密码：g5qa
5. 所有图片在实际运行时都转化为脸64x64的尺寸，最终的模型训练了200轮

### 文件说明

- 基本上和文件名相对应model_ckpt存放的是预训练的模型，直接运行test.py就可以使用了(前提是argments.py里面图片路径是正确的)。save_image是我之前跑模型时存下来看效果的图片结果。test_image是最后使用预训练模型得到的测试结果。

---

### 效果展示

**真实人脸**

![image](https://github.com/lzw-all-in/anime_person_translation/blob/master/test_image/0__person_real.jpg?raw=true)

**人脸转化的动漫脸**

![image](https://github.com/lzw-all-in/anime_person_translation/blob/master/test_image/0__anime_test.jpg?raw=true)



**真实动漫脸**

![image](https://github.com/lzw-all-in/anime_person_translation/blob/master/test_image/0__anime_real.jpg?raw=true)

**动漫脸转化的人脸**

![image](https://github.com/lzw-all-in/anime_person_translation/blob/master/test_image/0__person_test.jpg?raw=true)



## 自我评价

**感觉虽然动漫脸和人脸的转换的结果还是很不错，但是人物和动漫人物却没有很相似的成分(个人认为)，并且不知道是不是因为动漫的数据集女性偏多的原因所以所有男性人脸都转化到了女性的动漫脸上了╮(￣▽￣)╭**

另外欢迎有任何问题的可以给我提**issue**哦！就酱！∩__∩y ！