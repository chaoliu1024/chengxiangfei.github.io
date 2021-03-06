
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
 
 互信息度量特征t包含了类别c的信息量。如果特征t在类别c中的分布和所有文档中的分布完全相同，那么I（U;C）=0。如果一个
 特征能够完全确定一篇文档的类别，那么它的互信息也会达到最大。
 根据上述公式计算出每个特征t和类别c的互信息之后，我们可以把每个特征t的互信息值按照从大到小排序，然后选择前K个特征作为训练模型的特征词汇。
 
 #### 卡方统计
 在统计学中，卡方检验通过观察实际值与理论值得偏差来检验两个随机变量是否相互独立。具体的做法是，先假设两个变量
 是独立的，即原假设。然后看观察值与理论值直接的偏差程度，如果偏差程度足够小，我们就接受原假设，即两者是相互独立的，误差是由抽样或者测量带来的误差。
 如果偏差程度很大，我们就认为两者是相关的，否定原假设。
 
 在特征选择中，两个随机事件分别是特征是否出现![](http://latex.codecogs.com/gif.latex?e_t)
 和类别是否出现![](http://latex.codecogs.com/gif.latex?e_c)。假如特征t的出现与否，对类别c的判断毫无关系，
 我们就可以认为特征t和类别c相互独立，那么此时两者的卡方统计量应该接近于0，而当两者的卡方统计量很大时，则特征t对类别c的
 判断会有较大的影响。 
 我们可以根据下面公式计算![](http://latex.codecogs.com/gif.latex?e_t)
 和![](http://latex.codecogs.com/gif.latex?e_c)的卡方值，然后排序，选取最大的k的特征作为训练模型的特征空间。
 
 ![](http://latex.codecogs.com/gif.latex?X^2(D,t,c)=\sum_{e_t\in\\{0,1\\}}\sum_{e_c\in\\{0,1\\}}\frac{(N_{e_te_c}-E_{e_te_c})^2}{E_{e_te_c}})
 
 其中N是观察到的频率，E是期望频率。例如![](http://latex.codecogs.com/gif.latex?E_{11})是特征t和类别c同时出现的期望。
 
 ![](http://latex.codecogs.com/gif.latex?E_{11}=N*P(t)*P(c)=N*\frac{N_{11}+N_{10}}{N}*\frac{N_{11}+N_{01}}{N}) 

 ![](http://latex.codecogs.com/gif.latex?D_{11}=\frac{(N_{11}-E_{11})^2}{E_{11}}=\frac{(N_{00}*N_{11}-N_{10}*N_{01})^2}{N(N_{00}+N_{01})(N_{00}+N_{10})})

同样计算出![](http://latex.codecogs.com/gif.latex?E_{00})、![](http://latex.codecogs.com/gif.latex?E_{10})、![](http://latex.codecogs.com/gif.latex?E_{01})
和![](http://latex.codecogs.com/gif.latex?D_{00})、![](http://latex.codecogs.com/gif.latex?D_{10})、![](http://latex.codecogs.com/gif.latex?D_{01})带入到卡方公式中，
可以得到

![](http://latex.codecogs.com/gif.latex?X^2(D,t,c)=\frac{N(N_{11}*N_{00}-N_{10}*N_{01})^2}{(N_{11}+N_{01})(N_{11}+N_{10})(N_{01}+N_{00})(N_{10}+N_{00})})

卡方统计的一个缺点是会放大稀有词的显著性。假如有一个词在所有的特征中仅仅出现了2次，这2次都在类别c中，这个词就是统计显著的，但出现次数这么少的词对于分类是没有什么帮助的。

#### 信息增益

信息熵（information entropy）度量样板集合纯度最常用的一个指标。假定当前样本集合D中第k个类别所占的比例为![](http://latex.codecogs.com/gif.latex?p_k)(k=1,2,...,|Y|),
则样本D的信息熵定义为：

![](http://latex.codecogs.com/gif.latex?Ent(D)=-\sum_{k=1}^{|Y|}P_klog_2P_k)

Ent(D)的值越小，D的纯度越高。

假如离散的属性a可能的取值为 ![](http://latex.codecogs.com/gif.latex?\\{a^1,a^2,...,a^v\\})，现在使用a把样本D划分为v个子集，其中第v个子集包含了D中所有在特征a上取值为
![](http://latex.codecogs.com/gif.latex?a^v)的样本，记为![](http://latex.codecogs.com/gif.latex?D^v)，我们可以根据上面信息熵的定义计算出![](http://latex.codecogs.com/gif.latex?D^v)
的信息熵。再考虑到不同的样本子集包含的样本数不同，给每个子集一个权重![](http://latex.codecogs.com/gif.latex?|D^v|/|D|)（![](http://latex.codecogs.com/gif.latex?|D|)为样本D的个数），
即样本数越多的子集影响越大，可以计算出属性a对样本D的进行划分所得到的信息增益（information gain）

![](http://latex.codecogs.com/gif.latex?Gain(D,a)=Ent(D)-\sum_{v=1}^v\frac{|D^v|}{D}Ent(D^v))

一般而言，信息增益越大，则用a进行划分得到的纯度提升越大。因此，一般情况下，信息增益被用来做决策树的划分属性选择。

在文本特征选择中，特征t只有出现或者不出现{0,1}两种情况。因此，信息增益

![](http://latex.codecogs.com/gif.latex?Gain(D,t)=Ent(D)-\frac{|D^0|}{|D|}Ent(D^0)-\frac{|D^1|}{|D|}Ent(D^1))

此时，信息增益度量的是一个特征t对整个分类的贡献，不能度量对某个类别的贡献。因此，信息增益只能做全局的特征选择，
即所有的类都共用一套特征集合，不能针对某个类别选择本类别特有的特征集合。


上述的特征选择方法各有优缺点，比较多的实验证明卡方统计会有比较好的效果。但具体实验时是不是要做特征选择，选择哪种方法，特征集的大小取多少，还是要靠实验对比来决定。