
## 文本特征选择

文本的特征选择（feature selection）是从训练集所出现的所有词（terms）中选出一个子集，只用这个子集作为文本分类
的特征来训练分类器的过程。为啥要做特征选择呢？一、减少特征空间的维度，加快模型的训练速度和预测速度。
二、去掉对分类没什么帮助的噪声特征，提高分类准确度。下面介绍几种常用的特征选择方法。

#### 基于频率的特征选择方法

基于频率的特征选择方法，顾名思义就是选择某个类别里面出现最多的词作为特征。这里频率可以被定义为文档频率（Document Frequency， DF）
或者collection frequency。DF是指在类别c中包含特征t的文档数，更适用于Bernoulli model。collection frequency 则是指特征t在类别c中出现的次数,
适用于 multinomial model。

基于频率的方法只考虑一个词和在一个类别中出现的频率，因此会倾向于选择一些出现的次数很多但对分类没有什么贡献的通用词，
比如新闻中通常出现的时间、月份等。但当特征选择的够多（几千个）的时候，基于频率的方法也会有不错的表现。这是因为当选择的特征够多的时候，
那些重要的类别指示词也会被选择到特征中。

#### 互信息

互信息（Mutual Information, MI）度量两个事件集合之间的相关性。平均互信息的定义如下

![](http://latex.codecogs.com/gif.latex?I(X;Y)=\sum_{y\in{Y}}\sum_{x\in{X}}p(x,y)log\frac{p\(x,y\)}{p\(x\)p\(y\)})

在文本特征选择中，我们可以通过上述公式计算类别c和特征t的互信息，以此来度量一个特征的出现或缺失对做出正确的分类决策的贡献大小。
假设`U`是一个代表 ![](http://latex.codecogs.com/gif.latex?e_t=1)（包含特征t的文档）和 ![](http://latex.codecogs.com/gif.latex?e_t=0)
（不包含特征t的文档）的随机变量，C是一个代表![](http://latex.codecogs.com/gif.latex?e_c=1) (属于类别c的文档)
和（不属于类别c的文档）![](http://latex.codecogs.com/gif.latex?e_c=0)，那么文本特征选择公式就是：

![](http://latex.codecogs.com/gif.latex?I(U;C)=\sum_{e_t\in\\{1,0\\}}\sum_{e_c\in\\{1,0\\}}P(U=e_t,C=e_c)log_2\frac{p\(U=e_t,C=e_c\)}{p\(U=e_t\)P\(C=e_c\)} )

 根据大数定理，我们可以用频率去估计公式中的概率。
 另 ![](http://latex.codecogs.com/gif.latex?N_{**}) 表示当![](http://latex.codecogs.com/gif.latex?e_t) 和 ![](http://latex.codecogs.com/gif.latex?e_c)的值取对应的下标时的文档数。
 例如： ![](http://latex.codecogs.com/gif.latex?N_{10})表示包含特征t但不属于类别c的文档数；![](http://latex.codecogs.com/gif.latex?N_{11})表示包含特征t且属于类别c的文档数；
 以此类推。可以得到![](http://latex.codecogs.com/gif.latex?N=N_{10}+N_{11}+N_{01}+N_{00}),![](http://latex.codecogs.com/gif.latex?N_{1.}=N_{10}+N_{11}) 。
 根据大数定理，我们可以估计当U=1，且C=1时的概率，![](http://latex.codecogs.com/gif.latex?P(U=1,C=1)=N_{11}/N)。据此，我们可以把上面的互信息公式重写为：
 
 ![](http://latex.codecogs.com/gif.latex?I(U;C)=\frac{N_{11}}{N}log_2\frac{NN_{11}}{N_{1.}N_{.1}}+\frac{N_{01}}{N}log_2\frac{NN_{01}}{N_{0.}N_{.1}}+\frac{N_{10}}{N}log_2\frac{NN_{10}}{N_{1.}N_{.0}}+\frac{N_{00}}{N}log_2\frac{NN_{00}}{N_{0.}N_{.0}} )
 
 互信息度量了特征t包含了类别c的信息量。如果特征t在类别c中的分布和所有文档中的分布完全相同，那么I（U;C）=0。如果一个
 特征能够完全确定一篇文档的类别，那么它的互信息也会达到最大。
 根据上述公式计算出每个特征t和类别c的互信息之后，我们可以把每个特征t的互信息值按照从大到小排序，然后选择前K个特征作为训练模型的特征词汇。
 