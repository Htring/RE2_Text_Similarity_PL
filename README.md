## 背景

为了能够验证模型是否复现成功，并且我更偏向做中文的相关任务，对比开源项目:https://github.com/zhaogaofeng611/TextMatch在对应数据的复现结果，其在测试集上的ACC为：0.8391.
该论文pytorch版源码如下：https://github.com/alibaba-edu/simple-effective-text-matching-pytorch

程序讲解已发布在我的博客：[【NLP】文本匹配——Simple and Effective Text Matching with Richer Alignment Features(RE2)模型实现](https://blog.csdn.net/meiqi0538/article/details/124537692?spm=1001.2014.3001.5501)，需要了解如何实现的可以查看我的原码。

## RE2实现

沿袭以往的实现思路，程序依然分为一下模块：

- 数据处理模块dataloader
- 模型实现模块
- pytorch_lightning 训练封装模块
- 模型训练和使用模块

代码无须过多介绍，大致介绍一些比较有意思的程序。由于论文中很多模块会有多种处理方式，源码采用注册的方式去获取对应的模块。这种方式算是一种设计模块吧，值得学习一下。该种方式借助了一个装饰器函数，实现如下：

```python
def register(name=None, registry=None):
    """
    将某个函数获这某个类注册到某各地方，装饰器函数
    :param name: 注册的函数别名
    :param registry: 注册保存的对象
    :return: registered fun
    """
    def decorator(fn, registration_name=None):
        module_name = registration_name or fn.__name__
        if module_name in registry:
            raise LookupError(f"module {module_name} already registered.")
        registry[module_name] = fn
        return fn

    return lambda fn: decorator(fn, name)

```

源码对pytorch中的Linear进行了封装，增加了gelu激活函数，如下：

```python
class GeLU(nn.Module):
    __doc__ = """ gelu激活函数 """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class Linear(nn.Module):
    __doc__ = """ 改写的Linear层 """

    def __init__(self, in_features:int, out_features:int, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

除此之外还封装了一维卷积，如下：

```python
class Conv1d(nn.Module):
    __doc__ = """ 改写的一维卷积 """

    def __init__(self, in_channels, out_channels, kernel_sizes: Collection[int]):
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), 'only support odd kernel sizes'
        assert out_channels % len(kernel_sizes) == 0, 'out channels must be dividable by kernels'
        out_channels = out_channels // len(kernel_sizes)
        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(in_channels,
                             out_channels,
                             kernel_size,
                             padding=(kernel_size - 1) // 2)
            nn.init.normal_(conv.weight, std=math.sqrt(2. / (in_channels * kernel_size)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), GeLU()))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)

```

其他的内容，看看论文和源码应该没有多大问题了。

## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
   ![AIAS编程有道](https://s2.loli.net/2022/05/05/DS37LjhBQz2xyUJ.png)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)





