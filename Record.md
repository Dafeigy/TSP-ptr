# Neural Combinatorial Optimization with Reinforcement Learning

论文连接：[1611.09940.pdf (arxiv.org)](https://arxiv.org/pdf/1611.09940.pdf)。

## 引言

组合优化问题是一个计算机科学的基础问题。文章主要针对旅行商（TSP）问题进行研究，而TSP问题是一个NP-Hard的问题，实践中一般都是使用手工设计的启发式算法指导寻找问题的求解。但这些方法的鲁棒性不太行，如果问题的规模或者问题的条件发生了变化，就需要重新迭代，相比之下机器学习可以基于训练数据自动进行启发式的搜索。

大部分成功的机器学习算法都是监督学习，不太适用于组合优化问题。但是强化学习的方法也许可以解决组合优化问题，作者展示了即使使用了最优解作为标签的监督学习的算法也不如使用强化学习方法寻找到的解的性能。

作者将他们的方法称为神经组合优化（Neural Combinatorial Optimization），这是一种使用神经网络的强化学习方法解决组合优化问题的框架，作者使用了两种基于策略梯度的方法。第一种是RL预训练，使用了训练集以优化循环神经网络（RNN）作为策略输出的参数，目标是拟合期望奖励，在测试过程中策略是固定的，然后使用$\varepsilon -greedy$策略或采样进行交互；第二种方法是激活搜索，不包含预训练过程，使用随机的策略并优化RNN的参数已达到预期的目标，同时也保持跟踪最佳的搜索采样的解。这两种效果都很不错。

在二维平面约100个节点的欧拉距离图上，神经组合优化比使用监督学习的方法效果要好，并且接近最优解。作者还在KnpSack问题上进行了测试，发现在200个物品下算法依然展现良好性能。说明神经网络可以作为一个组合优化问题的求解工具，尤其是当启发式算法很难设计相关元素的时候能发挥良好作用。

## 先前工作

略

## TSP问题的神经网络架构

给定一个输入的图，用序列$s=\{x_i\}_{i=1}^{n}$表示$n$个城市的二维空间，其中$x_i\in\mathbb R^{2}$，我们期望找到一种点变换$\pi$，即旅行路线，使得只到访每个城市一次，并且使旅行路线最短。旅行路线所代表的点变换$\pi$的长度表示为：
$$
L(\pi|s)=||x_{\pi(n)}-x_{\pi}(1)||_2+\sum_{i=1}^{n=1}||x_{\pi(i)}-x_{\pi(i+1)}||_2
$$
其中$||\cdot||_2$是$L2$范数。期望优化随机策略$p(\pi|s)$，即给定点集$s$后以高概率输出短的旅程，以低的概率输出长的旅程。神经网络使用链式法则进行构造：
$$
p(\pi|s)=\prod_{i=1}^{n}p(\pi(i)|\pi(<i),s)
$$
然后使用softmax模块对上式右端表征概率。传统的seq2seq模型解决TSP时将TSP中的每一个城市表征为$\{1,2,\dots.n\}$，但是存在两个主要的问题：首先这种方式训练的神经网络并不能对问题规模超过$n$的实际场景求解；其次，需要拿到最优解才能使用条件对数似然优化seq2seq的模型参数。作者使用了一种利用非参数的softmax模块与注意力机制组合的指针网络以提升算法泛化能力，如图所示。

<img src="https://s2.loli.net/2023/02/08/ImzgWKM3Sl29xLn.png" alt="image-20230208211936761" style="zoom:50%;" />

整体的网络包含两个RNN模块——编码器和解码器，他们都由LSTM单元组成。编码器模块读取输入序列$s$，每次读取一个元素然后将其转换为潜在记忆状态（latent memory states）$\{enc_i\}_{i=1}^{n}$，其中$enc_i\in \mathbb R^d$。在$i$时刻输入到编码器模块的是一个点$x_i$二维坐标$d$维的embedding，embedding是对所有输入点$x_i$共享参数的一个线性转换。解码器模块同样保存其潜在记忆状态$\{dec_i\}_{i=1}^{n}$，其中$dec_i\in \mathbb R^d$且在每一个时间步$i$使用指针机制产生下一个需要到访的城市概率分布。当下一个到访城市被选定后，它将会被传入到解码器作为其下一个输入。解码器的初始输入是一个$d$维的可训练的向量，在图中以$<g>$表示。

注意力函数将每一个向量$q=dec_i \in \mathbb R^d$以及一系列的参考向量$ref=\{enc_1,\dots,enc_k\}$作为输入，并针对$k$个参考生成概率分布$A(ref,q)$。概率分布代表了模型在见到了序列$q$后指向的下一个参考点$r_i$的程度。

## 策略梯度优化

### 策略梯度与改进方法

在开始之前先要说说策略梯度是什么。它其实是MDP过程下的一个衍生的概念，给定一条状态-动作的交互轨迹$\tau=\{s_1,a_1,s_2,a_2,\dots\}$，参数为$\theta$的一个策略实体能复现这条轨迹的概率为：
$$
\begin{aligned}
p_\theta(\tau)=&p(s_1)p(a_1|s_1)p(s_2|s_1,a_1)p(a_2|s_2)\dots\\
=&p(s_1)\prod_{t=1}^{T}p_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)
\end{aligned}
$$
注意，策略只能控制通过观察状态选择动作这个过程，只有环境才能控制根据动作的选择作出响应。在每一次的交互过程中都会产生一个**奖励（reward）**$r_i$，这条轨迹上所有的reward加起来就是**回报**$R(\tau)$：
$$
R(\tau)=\sum_{\tau}r_i
$$
而回报的期望就是：
$$
\overline R_\theta(\tau)=\sum_{\tau}p_\theta(\tau)R(\tau)=\mathbb E_{\tau\sim p_\theta(\tau)}[R(\tau)]
$$
一般的MDP建模的主要目标之一就是最大化回报期望，因此一个可行的方法就是使用梯度上升法优化参数$\theta$使回报期望变大，而使用梯度上升的前提条件就是要计算优化式子的梯度：
$$
\nabla_\theta [\overline R_\theta(\tau)]=\nabla_\theta[\sum_{\tau}R(\tau)p_\theta(\tau)]
$$
注意到$R(\tau)$是和$\theta$无关的，因此可以改写成：
$$
\nabla_\theta [\overline R_\theta(\tau)]=\sum_{\tau}R(\tau)\nabla_\theta p_\theta(\tau)]
$$
注意上式中的$\nabla_\theta p_\theta(\tau)$满足如下式子：
$$
\begin{aligned}
\nabla_\theta p_\theta(\tau)&=\nabla_\theta [e^{\log p_\theta(\tau)}]\\
&=e^{\log p_\theta(\tau)}\nabla_\theta[\log p_\theta(\tau)]\\
&=p_\theta(\tau)\cdot\nabla_\theta[\log p_\theta(\tau)]
\end{aligned}
$$
所以回报期望的梯度可以表示为：
$$
\nabla_\theta[\overline R_\theta(\tau)]=\sum_\tau R(\tau)\nabla_\theta\log p_\theta(\tau)
$$
但实际上$\mathbb E_{\tau\sim p_\theta(\tau)}[R(\tau)]$是无法计算的，但可以从统计学的角度进行估计，具体做法就是采样$N$个$\tau$然后计算上式的值，把每一个值加起来求平均即可得到梯度的一个估计：
$$
\begin{aligned}\mathbb E_{\tau\sim p_\theta(\tau)}[R(\tau)\nabla_\theta\log p_\theta(\tau)]&\approx\frac{1}{N}\sum_{n=1}^{N}R(\tau_n)
\nabla_\theta\log p_\theta(\tau_n)\\
&=\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}R(\tau_n)\nabla_\theta \log p_\theta(a_t^n|s_t^n)
\end{aligned}
$$

上式就是所谓的策略梯度，我们将其标识为`PG`。

作者认为RL算法对组合优化问题有天然优势，因为这类问题的奖励机制很简单并且在实际部署时也可以用。网络用$\theta$表征，其优化的目标是希望给定输入图$s$后输出的旅程距离：
$$
J(\bold{\theta}|s)=\mathbb E_{\pi\sim p_{\theta}(\cdot|s)}L(\pi|s)
$$
训练过程中的输入图$s$是从$\mathcal S$采样得到的，训练的目标其实包含了这一部分，即$J(\theta|s)=\mathbb E_{s\sim \mathcal S}J(\theta|s)$。在使用策略梯度进行优化$\theta \leftarrow \eta+\nabla_\theta \overline R_\theta$时，可以使用一种叫做添加**基线（baseline）**的技巧。这种技巧是解决奖励函数设置为非负的情况出现的优化问题，如果对于任何状态$s_t$，无论执行什么动作都会产生非负的奖励，那么`PG`式就会等价为不管是什么动作，都要提升这个动作发生的概率，这显然有违“在观察环境后选取最优的动作”这一初衷。为了解决这个问题，我们可以把奖励减去$b$，从而保证$R(\tau)-b$这一项有正有负，当$R(\tau)>b$时就让$(s,a)$的概率上升，否则就减少其概率。

$b$的取值可以通过在交互过程中记录下的$R(\tau)$并对其求平均值将其作为期望，即$b\approx\mathbb E[R(\tau)]$，这是最简单的方式，然而还是会有采样不均匀的问题，具体来说就是**交互过程中可能得到一个好的结果，但并不代表整个过程中每一次交互的动作选择都是好的；同理交互过程可能得到一个坏结果，但并不代表整个过程中的每一次交互的动作选择都是坏的。**解决这个问题的方法就是为动作分配一个权重，以此确定哪些动作在某个状态下是最好的。一般来说，$b$是一个神经网络估计出来的，这个神经网络我们称为Critic网络。它可以是依赖状态的，我们称$R(\tau)-b$这一项为优势函数**（Advantage Function）**，优势函数在意的不是绝对的“好”结果，而是相对的“好”结果，即**相对优势**。

### 时序差分与蒙特卡洛方法

聊方法前，先明确下我们的核心目标：求解策略梯度。时序差法和蒙特卡洛方法其实就是求解方式的区别，借用磨菇书的图：

![img](https://datawhalechina.github.io/easy-rl/img/ch4/4.20.png)

可以看到二者的区别主要是计算与学习更新的时机。常用的REINFORCE算法实现获取每个步骤获得的奖励，然后计算每个步骤的未来总奖励$G_t$，然后将$G_t$代入：
$$
\nabla \overline R_\theta \approx\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}G_t^n\nabla \log \pi_\theta(a_t^n|s_t^n)
$$
解释一下$G_t^n$中的$n$表示一个回合中的从第$n$步到最后一步的轨迹的子集，$t$则是步骤的下标。一般来讲$G_t$是奖励的加权和，即：
$$
G_t=\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k\\=r_{t+1}+\gamma G_{t+1}
$$
REINFOCE算法需要产生一个回合的数据，即$\{(s_1,a_1,G_1),(s_2,a_2,G_2\},\cdots(s_T,a_T,G_T)\}$，然后针对每个动作计算梯度$\nabla \ln \pi(a_t|s_t,\theta)$，然后用NCEloss计算损失，将损失作为优化的目标进行优化，如图所示：

![img](https://datawhalechina.github.io/easy-rl/img/ch4/4.28.png)

回到原文，使用随机梯度下降对参数进行优化。上式的梯度可以使用REINFORCE算法进行表述：
$$
\nabla_\theta J(\theta|s)=\mathbb E_{\pi\sim p_{\theta}(\cdot|s)}[(L(\pi|s)-b(s))\nabla_{\theta}\log p_{\theta}(\pi|s)]
$$
其中$b(s)$是独立于策略$\pi$的基线函数，它通过测量期望的旅程长度以减少梯度的方差。

从分布$\mathcal S$采样得到的独立同分布的$B$张图$s_1,s_2,\dots,s_B$并对每一张图进行旅程的一次采样（比如说使用一个策略$\pi_i\sim p_\theta(\cdot|s_i)$）后，上面的式子使用蒙特卡洛采样进行近似：
$$
\nabla_{\theta}J(\theta)\approx\frac{1}{B}\sum_{i=1}^{B}(L(\pi_i)-b(s_i))\nabla_\theta \log p_\theta (\pi_i|s_i)
$$
而基线函数$b(s)$考虑到策略会随着训练变得更好，因此可以选择网络取得的平均奖励，它是随着时间指数型变化的。这个基线函数也有问题，它不能辨别出两个不同的输入图像。特别的对于一个复杂图，最优的策略$\pi^{\star}$可能会使得$L(\pi^{\star}|s)>b$，因为$b$是对一个batch里的所有图进行交互计算的。所以作者使用了critic网络为每一个$s$生成baseline，critic网络的参数用$\theta_v$表示，它将输出使用当前策略$p_\theta$对输入序列$s$判别后得到的最短路程。critic同样使用SGD进行训练优化，其优化目标是其输出$b_{\theta_v}(s)$与实际的策略采样得到的路程长度之差的平方，作为额外的优化项表述为：
$$
\mathcal L(\theta_v)=\frac{1}{B}\sum_{i=1}^{B}||b_{\theta_v(s_i)-L(\pi_i|s_i)}||_2^2
$$

## TSP问题下的Critic网络结构

Critic网络映射输入序列$s$到一个基线函数$b_{\theta_v}$由如下几个模块组成：1.一个LSTM编码器 2. 一个LSTM处理模块 3. 一个两层的ReLu 神经网络解码器。其编码器和指针网络的编码器结构相同，将输入序列$s$转换为潜在记忆状态以及隐藏太$h$。处理模块和[Vinyals](https://arxiv.org/abs/1506.03134)论文中的相似，对隐藏状态进行$\mathrm P$步计算，每一步都会通过使用glimpsing记忆对隐藏状态更新，并对glimpse函数的输出作为下一步处理的输入。在处理模块的最后，得到的隐藏状态会被随后通过两个分别含有d个和1个单元的全连接层编码成基线预测（单个标量）。作者提出的算法和A3C很相近，因为critic的预测和对一张图抽样得到的路程是优势函数的无偏估计。作者在多个CPU核上进行异步更新，但每一个核都处理一个迷你批次以获得更好的梯度估计，算法如下所示：

![image-20230208223443134](https://s2.loli.net/2023/02/08/Z7b86OzXyDcPUIg.png)

## 两种搜索策略

评估路径长度相对来说资源开销很小，因此推理阶段可以针对每张图的多种可选方法进行搜索并选择其中最好的一个，这个过程和在巨型的解空间搜索可行解很像，本文作者考虑了如下两种方法。

### 采样（Sampling）

就是简单的从多个的随机策略$p(\cdot|s)$中采样得到路程并选择最短的一个。和启发式方法不同，并不强迫模型在这个过程中采样不同的路程，然而我们可以通过一个temperature的超参数控制无参数的softmax的输出采样的密度（附录2）.该采样过程比贪心策略得到的提升更大，因为后者总是选择最大概率对应的索引。作者也考虑了对指针机制加入随机噪声的扰动并从修改后的策略中使用贪心策略，但这个没啥用。

### 激活搜索（Acitive Search）

可以在单次测试阶段输入$s$ 后最小化$\mathbb E_{\pi \sim p_{\theta}(\cdot|s)L(\pi|s)}$对随机策略$p_{\theta}$的参数进行精调，而不是对固定的模型进行采样或忽略采样过程中的解的奖励信息。这种方法对训练过的模型进行改进，和对为训练过的模型进行调优都由不错性能，前者被作者称为RL预训练激活搜索，后者称为激活搜索，因为模型是在单词测试中搜索备选解时更新参数的。激活搜索的算法流程如下所示：

![image-20230208225425577](https://s2.loli.net/2023/02/08/VgfIUdwE2Ha1JM8.png)

和Critic网络的更新算法很相似，但是激活搜索对单次测试时的备选解$\pi_1,\dots,\pi_B\sim p_\theta(\cdot|s)$进行了蒙特卡洛采样。它采取了一个指数型变换的平均基线，这和critic不同，因为它不需要对输入之间进行区分。值得注意的一点是，强化学习训练尽管不需要监督，它仍然是需要训练数据的，并且其泛化能力和数据分布非常相关。然而激活搜索是独立分布的，由于对城市进行了编码成序列，作者就在将输入序列输入金指针网络前将其打乱，这能增加采样过程的随机性并为激活搜索带来性能提升。

## 实验结果

略

## 其他问题的泛化

略

## 结论

略

## 附录

### Pointing Mechanism

这部分的计算是由两个注意力矩阵$W_{ref},W_q\in\mathbb R^{d\times d}$以及一个注意力矢量$v\in\mathbb R^{d}$表示的：
$$
u_i=\begin{cases}
\begin{aligned}
&v^{\top}\cdot\tanh (W_{ref}\cdot r_i + W_q \cdot q) & i\neq \pi(j)\ \forall j<i\\
&-\infty& otherwise
\end{aligned}
\end{cases}
$$

$$
A(ref,q;W_{ref},W_q,v)=softmax(u)
$$

指针网络在第$j$步时会依照下式输出下一个访问的点的概率分布：
$$
p(\pi(j)|\pi(<j),s)=A(enc_i,dec_j)
$$
将已经访问过的城市点的logits值设为$-\infty$，确保模型只会输出符合TSP要求的点。

### Attending Mechanism

Glimpse函数$G(ref,q)$和注意力函数$A$采取相同的输入，他的计算表征由$W_{ref}^{g},W_q^g \in \mathbb R^{d\times d}$和$v^g \in \mathbb R^{d}$完成。依照如下式子进行计算：
$$
p=A(ref,q;W_{req}^g,W_q^g,v^g)\\
G(ref,q;W_{ref}^g,W_q^g,v^g)=\sum_{i=1}^{k}r_i p_i
$$
Glimpse函数是用注意力概率计算参考向量权重的线性组合，它同样可以在相同的参考集上运用多次：
$$
g_0=q\\
g_l=G(ref,q_{l-1};W_{ref}^g,W_q^g,v^g)
$$
最终，$g_l$向量将会传递到注意力函数$A(ref,g_l;W_{ref},W_q,v)$以产生pointing机制。我们观测到对于相同的模型参数多次使用glimpse和只使用一次相比并不能有效加快训练并提高训练结果。

### softmax temperature

我们将注意力函数修改为如下式子：
$$
A(ref,q,T;W_{ref},W_q,v)=softmax(u/T)
$$
其中的$T$是*温度超参数*，并且在训练阶段设置为$T=1$。当$T>1$时，$A(ref,q)$的分布会变得平缓些，因此可以避免模型出现过度置信的情况。

### logit clipping

修改注意力机制为如下式子：
$$
A(ref,q;W_{ref},W_q,v)=softmax(C \tanh(u))
$$
其中，$C$是控制logits值得超参数并借此控制$A(ref,q)$的熵。

