---
citekey: 2022DeepNeuralNetworkFusionGraph
Type: conferencePaper
Title: 'Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning (GAMF)'
Author: 'Chang Liu, Chenfei Lou, Runzhong Wang, Alan Yuhan Xi, Li Shen, Junchi Yan'
Journal: ''
Date: 2022-06-28
DOI:  
---

# Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning (GAMF)

## One-sentence summary
GAM**F将模型融合问题表述为图形匹配任务**，考虑了模型权重的二阶相似性，而不是之前的工作仅仅将模型融合表述为一个线性赋值问题。针对问题规模的扩大和多模型的一致性问题，GAMF提出了一种高效的基于分级赋值的模型融合方法，以保持一致性的方式迭代更新匹配结果。

## Metadata

>[!info] Info
>- **Title**: Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning (GAMF) 
>- **Author**: Chang Liu, Chenfei Lou, Runzhong Wang, Alan Yuhan Xi, Li Shen, Junchi Yan
>- **Year**: 2022-06-28
>- **Bibliography**: Liu, C., Lou, C., Wang, R., Xi, A. Y., Shen, L., & Yan, J. (2022). Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning (GAMF). _Proceedings of the 39th International Conference on Machine Learning_, 13857–13869. [https://proceedings.mlr.press/v162/liu22k.html](https://proceedings.mlr.press/v162/liu22k.html)

>[!abstract] Abstract
> Model fusion without accessing training data in machine learning has attracted increasing interest due to the practical resource-saving and data privacy issues. During the training process, the neural weights of each model can be randomly permuted, and we have to align the channels of each layer before fusing them. Regrading the channels as nodes and weights as edges, aligning the channels to maximize weight similarity is a challenging NP-hard assignment problem. Due to its quadratic assignment nature, we formulate the model fusion problem as a graph matching task, considering the second-order similarity of model weights instead of previous work merely formulating model fusion as a linear assignment problem. For the rising problem scale and multi-model consistency issues, we propose an efficient graduated assignment-based model fusion method, dubbed GAMF, which iteratively updates the matchings in a consistency-maintaining manner. We apply GAMF to tackle the compact model ensemble task and federated learning task on MNIST, CIFAR-10, CIFAR-100, and Tiny-Imagenet. The performance shows the efficacy of our GAMF compared to state-of-the-art baselines.

> [!example] Files and Links 

>- **DOI**: 
>- **Url**: https://proceedings.mlr.press/v162/liu22k.html
>- **PDF**: [2022_Liu et al_Deep Neural Network Fusion via Graph Matching with Applications to Model.pdf](file://C:\Users\10437\OneDrive%20-%20sjtu.edu.cn\Zotero\storage\2022PMLR\2022_Liu%20et%20al_Deep%20Neural%20Network%20Fusion%20via%20Graph%20Matching%20with%20Applications%20to%20Model.pdf)
>- **Zotero**: [2022_Liu et al_Deep Neural Network Fusion via Graph Matching with Applications to Model.pdf](zotero://select/library/items/89BV93QA)

> [!tip] Zotero Tags
>- **Keywords**: #unread, #CCF-A

## Annotations


--- 
# Obsidian Notes
由于实际的资源节省和数据隐私问题，机器学习中无需访问训练数据的模型融合引起了越来越多的兴趣。在训练过程中，每个模型的神经权重可以随机排列，并且我们必须在融合之前对齐每层的通道。将通道重新分级为节点，将权重重新分级为边缘，**对齐通道以最大化权重相似性是一个具有挑战性的 NP 困难分配问题**。由于其二次分配性质（quadratic assignment nature），我们将模型融合问题表述为图匹配任务，考虑模型权重的二阶相似性，而不是以前的工作仅仅将模型融合表述为线性分配问题。针对不断上升的问题规模和多模型一致性问题，我们提出了一种高效的基于分级分配的模型融合方法，称为 GAMF，该方法以保持一致性的方式迭代更新匹配。我们应用 GAMF 来处理 MNIST、CIFAR10、CIFAR-100 和 Tiny-Imagenet 上的紧凑模型集成任务和联邦学习任务。该性能显示了我们的 GAMF 与最先进的基线相比的有效性。

## 辅助资料


## What can we learn
- 原来FL中的模型聚合（model aggregation）其实在model fusion的说法中已有研究，可以看一看！

## Q1 论文试图解决什么问题？

如果我们有两个或多个独立训练的神经网络，我们应该如何以最准确的方式利用它们？ (Utans, 1996) 提出模型融合问题，旨在将多个神经网络融合成一个网络，而不访问训练数据。与传统的基于预测的模型集成相比，将多个网络融合为一个的优点是节省内存和推理时间，因为预测集成（prediction ensemble）需要维护所有单独的模型。此外，它还可以应用于privacy-intensive联邦学习（FL），其中如何有效地聚合所有本地训练的模型仍然是开放的问题。

假设不同网络的通道不需要任何对齐，那么普通策略是简单地平均权重。然而，由于深度神经网络的随机性和正交不变性，来自不同网络的通道总是随机排列的（randomly permuted）。之前的研究（Singh & Jaggi，2020）已经证明了模型融合中通道不对齐的危害性，因为网络的有效组件将相互干扰和抵消。

![500](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230703122018.png)
## Q2 这是否是一个新的问题？

## Q3 这篇文章要验证一个什么科学假设？

## Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

### 多模型解决方案
现有的多模型解决方案相对启发式，例如：
- OTFusion (Singh & Jaggi, 2020) 只是按顺序合并所有模型
- FedMA (Wang et al., 2020a) 在每个聚合步骤按顺序选择一个锚点
- FedSpa (Huang et al., 2022b) 和 DisPFL (Dai et al., 2022) ）通过稀疏训练提取子模型，将多个局部模型融合在低维子空间中（Liu et al., 2021b; 2022），
- Wang et al. (2022)采用分布式鲁棒优化方法来融合多个模型
- 相比之下，多图匹配（MGM）算法（Yan et al., 2016a; Jiang et al., 2021; Wang et al., 2020b; Leonardos et al., 2017）是在循环一致性（cycle consistency）的意义上开发的，该算法确保涉及任何第三个图的匹配不会违反两个图的匹配。因此，MGM 算法可以确保多个模型的全局对齐。

尽管上面讨论了吸引人的特性，现有的图匹配方法（Gold & Rangarajan, 1996; Cho et al., 2010; Jiang et al., 2021; Zhou & Torre, 2016; Wang et al., 2020b），包括状态-最先进的商业解决方案 GUROBI (Optimization, 2020) 不能轻易应用于神经网络模型融合，因为 $O((d_{\Sigma})^{4})$大小的亲和力张量（affinity tensor）引入了内存负担，特别是对于融合多个模型。这里的$d_Σ$是神经网络的通道总和，对于现代神经网络来说可以达到几千个。幸运的是，我们将证明亲和力张量包含某些稀疏模式，这为通过受经典作品启发的分级分配技术（graduated assignment technique）（Gold & Rangarajan，1996）进行更具成本效益的开发（more cost-effective exploitation）留下了空间。

### Model Fusion
在本文中，具体指的是在不访问训练数据的情况下将两个预训练网络合并为一个网络，这比对每个网络的预测进行平均的传统模型集成（Breiman，1996；Wolpert，1992；Schapire，1999）更有效。 (Leontev et al., 2020) 将模型融合表示为线性分配并近似求解。然而，这些方法的假设非常严格，例如，原始模型必须共享一部分训练历史（Smith & Gashler，2017；Utans，1996）或依赖 SGD 进行周期性平均（Malinovskiy et al.，2020） ）。遗憾的是，他们无法确保开销平均（Leontev et al., 2020）。**请注意，有大量关于多模型推理阶段模型计算和结果融合的工作（Du et al., 2017），而我们论文中的融合是指训练后的模型参数的融合。**

与我们最相关的工作是 OTFusion（Singh & Jaggi，2020），它注意到模型融合内部的权重对齐性质（weights alignment nature）。它将模型融合表述为线性分配问题（linear assignment problem），并通过 Wasserstein 重心（Wasserstein barycenters）解决它，这是第一个用于改进普通平均（vanilla averaging）的模型融合工作，然而，它通过忽略权重的二阶相似性（second-order similarity）而使问题退化。与 OTFusion 相比，GAMF 解决了二次模型融合问题（quadratic model fusion problem）

### Federated Learning
联合学习是一种允许本地客户协作训练共享全球模型的范例。在其标准管道中，每个本地客户端使用自己的数据集训练本地模型，全局服务器收集所有本地模型并将它们合并到共享的全局模型中。最近，研究人员（Kairouz 等人，2019；Wang 等人，2021a）深入探讨了多种提高 FL 性能的方法（Fraboni 等人，2021；Chen 等人，2020；He 等人， 2020a；Wu 和Gong，2020；Dinh 等，2020；Deng 等，2020；Peterson 等，2019；Liu 等，2021a；Acar 等，2020；He 等，2021a； b；Huang 等人，2022a；Yu 等人，2021）。提高联邦学习性能的努力可以概括为两类：
- 改进局部优化器（local optimizer），例如：
	- FedAvg (McMahan et al., 2017)
	- FedProx (Li et al., 2018)
	- SCAFFOLD (Karimireddy et al., 2020) 
	- Moon (Li et al., 2021) 
- 增强服务器中的模型聚合，例如：
	- FedFTG（Zhang 等人，2022）
	- FedMA（Wang 等人，2020a）及其前身 PFNM（Yurochkin 等人，2019）
	- Moon 是最先进的联邦学习算法，它在本地客户的训练周期中增加了对比损失。

**我们强调，所有这些本地训练类型方法都与我们的 GAMF 正交，它可以作为增强其性能的强大插件。另一方面，FedMA 使用与 OTFusion 类似的分配公式（assignment formulation），并提出了一种迭代方法，这是与我们最相关的联邦学习算法。我们的 GAMF 与 FedMA 一致，因为我们都专注于不同本地模型的协调。**

### Graph Matching
图匹配的目的是通过考虑节点特征和边属性来找到节点对应关系，这在其一般形式中被称为 NP-hard (Loiola et al., 2007)。**通过将神经通道和参数权重视为节点和边缘属性，我们可以将模型融合问题表述为图匹配任务，旨在找到模型参数的最佳对应关系**。经典方法主要采用不同的优化启发式，包括：
- 随机游走（random walk）（Cho et al., 2010）
- 光谱匹配（spectral matching）（Leordeanu & Hebert, 2005）
- 路径跟踪算法（path-following algorithm）（Zhou & Torre, 2016）
- 分级作业（graduated assignment）（Gold & Rangarajan, 1996）
- SDP松弛（SDP relaxation）（Schellewald & Schn ̈ orr, 2005）

近年来，深度图匹配已成为一种新兴范式（Wang et al., 2019; 2021b; Yu et al., 2020; Rolı ́nek et al. .，2020；Liu 等人，2020）。读者可以参考这些调查来详细回顾传统图匹配（Yan 等人，2016b）和深度 GM（Yan 等人，2020）。

然而，标准图匹配通常处理具有数十个关键点的一般图像（Yan et al., 2020），而模型融合的可扩展性在现代网络中至少是数千个神经通道。在我们的尝试中，即使是商业求解器也无法有效地处理如此大的规模，例如GUROBI（优化，2020）。

## Q5 论文中提到的解决方案之关键是什么？
在本文中，我们采用一种网络模型融合方法的图匹配公式：分级分配模型融合（GAMF）。为了可扩展性，我们在内存高效的切片和扫描过程下开发了一种新的分级分配算法，使得通过图匹配融合深度神经网络成为可能。对于多模型融合，我们通过开发循环一致的 MGM 算法，提出了 GAMF 的多模型版本，其灵感来自（Wang & et al, 2020；Sol ́ e-Ribalta & Serratosa, 2013）。

GAMF 在两个重要应用中得到了验证：紧凑模型集成（compact model ensemble）和联邦学习。在第一种情况下，我们遵循 OTFusion 中的设置，GAMF 在 MNIST (LeCun, 1998) 和 CIFAR10 (Krizhevsky et al., 2009) 上均优于 OTFusion。在联邦学习中，我们在流行的开源框架FedML上进行了实验（He et al., 2020b）。与最先进的 FL 算法相比，GAMF 在 CIFAR-10、CIFAR-100 (Krizhevsky et al., 2009) 和 Tiny-Imagenet1 上收敛得更快。

### 模型融合的图匹配公式
在网络训练过程中，由于随机梯度下降的随机性以及不同模型训练集的差异，通道的排列可能会在不同模型之间进行混洗，这需要模型融合的通道对齐，如（Smith & Gashler， 2017；Leontev 等人，2020；Malinovskiy 等人，2020），早期工作可追溯到（Utans，1996）。在本文中，我们证明图匹配的公式自然地就可以用于模型融合。

为了简单起见，我们讨论关于融合两个全连接网络和两个没有bias的隐藏层的 GM 公式（见图 2a）：
$${\bf x}_{1}=\delta({\bf x}_{0}{\bf W}_{1});{\bf x}_{2}=\delta({\bf x}_{1}{\bf W}_{2});{\bf x}_{3}=\delta({\bf x}_{2}{\bf W}_{3}) \tag{1}$$
![500](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230703122103.png)
其中$x_0$是输入，$x_3$是输出，每个包含n个数据点。 $x_i ∈ R^{n×d_i}$ 表示第 i 层之后的维度为 di 的数据，$W_i ∈ R^{d_{i−1}×d_i}$ 是神经网络的权重矩阵。 δ 代表激活（例如 ReLU）。

下面我们通过上标加括号来区分不同的模型。模型融合就是找到通道的合理排列（即shuffle），相当于排列 $\mathsf{W}_{1}^{\left(1\right)},\mathsf{W}_{2}^{\left(1\right)},\mathsf{W}_{3}^{\left(1\right)}$来拟合$\mathsf{W}_{1}^{\left(2\right)},\mathsf{W}_{2}^{\left(2\right)},\mathsf{W}_{3}^{\left(2\right)}$，并且可以通过图匹配来制定和处理。当融合两个具有2个隐藏层的全连接网络时，我们有以下图匹配公式（P的其他元素为0，因为跨层匹配没有意义，因此P的结构非常稀疏）：
$$\operatorname*{max}_{\mathbf{P}} \sum_{i=0}^{d_{\Sigma}-1}\sum_{j=0}^{d_{\Sigma}-1}\sum_{a=0}^{d_{\Sigma}-1}\sum_{b=0}^{d_{\Sigma}-1}{\bf P}_{[i,j]}{\bf K}_{[i,j,a,b]}{\bf P}_{[a,b]} \tag{2}$$
$$s.t.\:{\bf P}_{0}={\bf I};{\bf P}_{3}={\bf I};$$$$\forall j\sum_{i=0}^{d_{1}-1}{\bf P}_{1[i,j]}=1,\forall i\sum_{j=0}^{d_{1}-1}{\bf P}_{1[i,j]}=1,\forall j\sum_{i=0}^{d_{2}-1}{\bf P}_{2[i,j]}=1,\forall i\sum_{j=0}^{d_{2}-1}{\bf P}_{2[i,j]}=1$$
其中：
- 我们的索引从0开始
- $d_Σ=d_0+d_1+d_2+d_3$
- P0、P1、P2、P3的定义来自图2b。

请注意，对于等式(2)：P对两个隐藏层的排列进行编码，输入/输出层的通道不需要排列（见图2a）。这些约束确保同一层内通道之间的一对一映射。在图匹配公式中，$K ∈ R^{d_Σ×d_Σ×d_Σ×d_Σ}$是一个 4 维亲和力张量（affinity tensor），其元素$K_{[i,j,a,b]}$ 测量边 (i, a) 和 (j, b) 之间的亲和力，这是模型融合问题中权重矩阵元素之间的相似度。我们采用 GM 方法广泛应用的高斯核作为相似性度量（Yan et al., 2016a; Cho et al., 2010）：
$$K_{[i,j,a,b]}=\exp\left(-{\frac{\left|\left|\mathbf{e}_{i,a}-\mathbf{e}_{j,b}\right|\right|_{F}^{2}}{\sigma}}\right)$$
其中 $e_{i,a}$ 表示边 (i, a) 对应的网络权重，σ 是可配置的超参数。

虽然图2a所示的模型融合的公式是基本的全连接层，但在之前的工作中（Singh & Jaggi，2020；Wang et al.，2020a）已经证明，包括卷积层在内的几个常用层也可以是融合在这样的公式中。

### Intrinsic Sparsity of the Model Fusion Task
方程 (2) 的问题在于：
- 已知的 (Yan et al., 2020) NP-hard 
-  Lawler 二次分配问题的特殊情况 (Lawler, 1963)，其内存成本在一般情况下可以是  ${O}\bigl(\bigl(d_{\boldsymbol{\Sigma}}\bigr)^{4}\bigr)$，其中 dΣ 是所有层的通道总和。

对于模型融合中考虑的具有数千个通道的深度网络，这似乎很棘手，事实上我们根据经验发现商业求解器 GUROBI（Optimization，2020）无法处理这样的内存负担。

幸运的是，值得注意的是，虽然矩阵 K 和 P 的尺度（scales）非常大，但这两个矩阵的分量却相对稀疏。如图2b所示，对于网络模型融合来说，重要的是要意识到：$P_0, P_{1,}P_{2,}P_3$之外的分量为0，因为跨层匹配是没有意义的。

假设层数为l，则最优成本上限为$O\bigl(\Sigma\bigl({\frac{d_{\Sigma}}{l}}\bigr)^{4}\bigl)= O\bigl(\frac{1}{l^{3}}(d_{\Sigma})^{4}\bigr)$，对于一个10 层网络，成本为原始${O}\bigl(\bigl(d_{\boldsymbol{\Sigma}}\bigr)^{4}\bigr)$的千分之一 。因此，我们的目标是探索稀疏结构来有效地解决问题。

### Graduated Assignment for Model Fusion
基于上述对问题固有稀疏性的分析，我们采用经典的graduated assignment(Gold & Rangarajan, 1996)，对方程(2)的一阶泰勒级数进行反复优化，然后将其映射到具有逐渐收紧约束的（宽松的）可行空间。我们提出的算法，namely **G**raduated **A**ssignment for **M**odel **F**usion (**GAMF**)，利用了 K 和 P 的底层稀疏结构，从而解决了计算和内存负担，并且可以在实验中扩展到像 VGG11 这样的网络。

#### Graduated Assignment Model Fusion
给定一个可行的置换矩阵 $P^0$，我们将等式（2）中的目标记作J。 Inspired by (Gold & Rangarajan, 1996)，可以通过其泰勒级数重写：
$$J=\sum_{i=0}^{d_{\Sigma}-1}\sum_{j=0 }^{d_{\Sigma}-1}\sum_{a=0}^{d_{\Sigma}-1}\sum_{b=0}^{d_{\Sigma}-1}\mathbf{P}_{[i,j]}^{0}\mathbf{K}_{[i,j,a,b]}\mathbf{P}_{[a,b]}^{0}+\sum_{i=0}^{d_{\Sigma}-1}\sum_{j=0 }^{d_{\Sigma}-1}R_{[i,j]}(P_{[i,j]}-P_{[i,j]}^{0)+...}\tag{4}$$
其中：
$$R_{[i,j]}=\left.{\frac{\partial J}{\partial\mathbf{P}}}\right|_{\mathbf{P}=\mathbf{P}^{0}}=\sum_{a=0}^{d_{\Sigma}-1}\sum_{b=0}^{d_{\Sigma}-1}\mathbf{K}_{[i,j,a,b]}\mathbf{P}_{[a,b]}^{0}$$
仅考虑方程（2）的零阶和一阶泰勒级数。它等于最大化以下目标，因为所有其他元素都是常数：
$$\operatorname*{max}_{\mathbf{P}}\sum_{i=0}^{d_{\Sigma}-1}\sum_{j=0 }^{d_{\Sigma}-1}R_{[i,j]}P_{[i,j]} \quad\quad s.t. constrains\ in\ Eq.(2)$$
上述约束优化问题是一个线性分配问题，可以通过Hungarian算法（Kuhn，1955）将实值方阵R投影到0/1置换矩阵，在多项式时间内求解至最优。此外，通过 Sinkhorn 算法可以实现松弛投影，该算法首先对正则化因子 τ 进行归一化：$S = exp(R/τ)$，然后交替进行行和列归一化：
$$\mathbf{D}_{r}=\mathrm{diag}(\mathbf{S1}),\mathbf{S}=\mathbf{D}_{r}^{-1}\mathbf{S};\mathbf{D}_{c}=\mathrm{diag}(\mathbf{S}^{\mathsf{T}}\mathbf{1}),\mathbf{S}=\mathbf{SD}_{c}^{-1}$$
其中diag(·)表示由向量构建对角矩阵，1是元素全为1的列向量。当上述算法收敛时，S是一个元素连续且所有行和列相加的双随机矩阵到 1，这是置换矩阵的宽松版本。 τ 控制了 Sinkhorn 算法和Hungarian算法之间的差距：给定较小的 τ ，Sinkhorn 算法的结果变得更接近Hungarian算法，但代价是需要更多的迭代才能收敛。

分级分配算法的工作原理是首先随机初始化$P^0$，计算R，然后通过Sinkhorn算法计算S，并重复这些步骤。 Sinkhorn 算法的 τ 在每一步都逐渐缩小。该框架由（Gold & Rangarajan，1996；Wang & et al，2020）开发并采用，以有效解决图匹配等难题组合问题。

#### Efficient Adaption to Model Fusion
通过分析图2中的K、P，这两个矩阵包含稀疏结构，因为在计算式（2）中的objective score时，只有来自同一层的节点和边才有效。为了节省内存，我们引入了用于模型融合的内存高效分级分配算法，如下所示。对于图 2 中的示例，两个神经网络的权重分别记作：$\mathsf{W}_{1}^{\left(1\right)},\mathsf{W}_{2}^{\left(1\right)},\mathsf{W}_{3}^{\left(1\right)}$和$\mathsf{W}_{1}^{\left(2\right)},\mathsf{W}_{2}^{\left(2\right)},\mathsf{W}_{3}^{\left(2\right)}$

第 i 层的 $R_i$ 可以通过融合第 i − 1 层和 i + 1 层的信息来获得。它可以被视为用内存效率更高的slide-and-scan过程代替 R 原本的计算过程。我们的两个模型的 GAMF 在 Alg 1 中进行了总结。

![600](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230703194228.png)

理论上，GAMF 的空间复杂度仅为 $O(c * l * d^2_{max})$，因为我们使用迭代的slice-and-scan过程，其中 c 是客户端数量，l 是层数，$d_{max}$ 是所有层的最大通道数。它使 GAMF 成为一种节省内存的方法，不存在超出内存或意外崩溃的风险。

#### Fusing Multiple Models by Multi-Graph Matching
多图匹配问题（Yan et al., 2016a；Jiang et al., 2021；Shi et al., 2016）是经典两图匹配问题的自然扩展，其中需要联合处理两个以上的图，通过强制多图正则化（namely循环一致性）。循环一致性特性确保排列结果不违反任何两对图，这对模型融合问题很有吸引力。
（🧐好疑惑哦）

在本文中，我们扩展了 Alg  1 与 GAMGM (Wang & et al, 2020)以融合多个模型：
Alg 1 中的$P_i$被替换为 $U^{(k)}_i$ ，其中$U_i^{(1)}U_i^{(2)⊤}$表示模型 1 和模型 2 的第 i 层的排列。来自多个图的信息在特殊情况下也以相同的方式融合两个图的条件。Alg 2中描述了多个模型的模型融合算法。另外，在多模型融合中，不同模型的权重设置相同，因为我们假设每个模型的重要性相同。

![600](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230703194929.png)

#### Connections and Differences with OTFusion
将我们的方法与 OTFusion (Singh & Jaggi, 2020) 进行比较，在计算目标分数的一阶泰勒级数时，我们的流程中也会遇到类似的步骤。然而，OTFusion 的主要缺点是二阶信息未得到充分利用，并且 OTFusion 可以被视为我们算法的一个特殊且简单的情况，只需进行一次迭代。相反，我们能够通过优化方程中的二次目标得分来处理图匹配中的二阶信息。 (2)采用多步流水线。同时，我们介绍了可以轻松集成到融合多个神经网络的常见场景中的多图匹配算法。此外，我们成功地将我们的模型融合方法 GAMF 应用于联邦学习，这是（Singh & Jaggi，2020）中未考虑的。更重要的是，我们发现 OTFusion 在 4.3 节的联邦学习实验中失败了。

总的来说，我们在现有工作之外的贡献有三方面：
1. 首先，我们将模型融合问题从一阶推广到二阶，将其表述为 GM 问题，并实现性能增益。
2. 其次，我们将 GAMF 应用于新兴的 FL 领域，并表明 GAMF 也可以处理 FL，但 OTFusion 不能很好地工作。
3. 第三，我们的工作将图匹配扩展到经典图像匹配之外的模型融合领域，这是一个不同的问题，特别是问题规模从数十个关键点上升到数千个点。为了解决可扩展性问题，我们设计了具有新颖的三层切片窗口方案的 GAMF（three-layer slicing-window scheme）。


## Q6 论文中的实验是如何设计的？

### Compact Model Ensemble Experiments
在这个实验中，我们的目的是测试模型融合的性能以生成紧凑的集成模型。集成模型是所有局部模型的组合，传统的方法是对它们的预测（最后一层的输出）进行平均。预测集成总能达到更好的性能，但代价是维持所有局部模型的参数。

继 OTFusion (Singh & Jaggi, 2020) 之后，我们希望找到一种更高效、更紧凑的模型集成方式，**即只维护一个模型而不是维护所有模型**。我们在 MNIST 和 CIFAR-10 中进行紧凑模型集成实验，设置与 OTFusion 相同。

#### Experiments on MNIST
我们尝试将两个简单网络合并（with the homogeneous and heterogeneous data partition），如图 3 所示。我们显示了不同集成比例的结果，因为第二个模型的比例从 0.0 增加到 1.0。模型融合一次性完成，无需任何进一步的微调。如图 3 所示，我们可以看到：
- 在同质数据划分设置中，集成的性能并不显着，因为两个模型太相似且训练有素，both GAMF and GUROBI are close to the prediction ensemble。
- 在异构数据划分中，GAMF 取得了良好的性能，甚至优于prediction ensemble。

![600](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230704101257.png)

#### Experiments on CIFAR-10
我们测试了标准图像分类数据集中现代神经网络的融合。我们在这些实验中选择 CIFAR-10 和 LeNet/VGG11 网络。对于每个骨干神经网络，我们使用 2/4 模型和同质/异质数据划分来测试模型融合性能。请注意，从现在开始我们不再使用基线 GUROBI，因为 LeNet 和 VGG11 太大，GUROBI 无法处理。结果如表 1 和表 2 所示。与 (Singh & Jaggi, 2020) 类似，我们报告了融合模型的one-shot结果和finetune结果。

在 LeNet 实验中，GAMF 在所有设置中都优于 OTFusion 和 Vanilla，尤其是在多模型设置中。虽然所有方法的one-shot性能都很差，但是GAMF仍然可以达到原始模型70%左右的准确率
GAMF 的微调性能优于原始模型，但比预测集成稍差。融合模型的表现通常比预测集成差（Singh & Jaggi，2020）。 GAMF的融合模型在精度方面稍差一些，但它具有2倍或4倍的效率。
![](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230704103804.png)


在 VGG11 实验中，OTFusion 表现在两个极端。 OTFusion 在同构数据分区中表现良好，在多模型设置中甚至优于 GAMF，但在异构数据分区中无法收敛。除了最后一行之外，我们的 GAMF 表现更加稳定。**具有异构数据分区的 4 个模型的设置对于模型融合方法来说似乎太难了，因为所有方法都不能很好地工作。此设置中的微调更像是从头开始重新训练网络。**==（一个可以改进的工作点！！）==

![](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230704104742.png)

#### Comparison of first-order and second-order similarity
在GAMF的设计中，我们考虑了二阶相似性，而不是现有的只关心一阶相似性的工作，例如OTFusion和FedMA。为了证明二阶相似性的有效性，我们在 CIFAR-10 和 VGG11 上进行了超紧凑的集成实验。我们对OTFusion和GAMF求解的解进行随机扰动，并绘制图4中相似度-准确度的散点图。**我们可以看到，GAMF的曲线比OTFusion的曲线更加平滑和相关。因此，我们认为二阶相似度在有效模型融合的意义上优于一阶相似度。**（🤔不太懂）

![600](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230704105438.png)

### Federated Learning Experiments
我们使用FedML框架（He et al., 2020b）并在CIFAR-10/100和Tiny-Imagenet上进行实验。除了基线之外，我们将在整个数据集上训练出来的准确性报告为“Entire”，这可以作为上限。

#### Experiments on the CIFAR-10
图 5a 和 5b 分别显示了 5 个客户端和 10 个客户端的结果。 GAMF 和 Moon 达到相似的最终精度，但 GAMF 比 Moon 收敛得更快。 **GAMF 没有超越 Moon 是合理的，因为 Moon 和 GAMF 分别侧重于联邦学习的不同部分进行改进。 Moon旨在改进本地训练，而GAMF则侧重于通信中的模型融合。**
![](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230704105653.png)

对于与 GAMF 关注相同部分的竞争方法 OTFusion 和 FedMA 来说，GAMF 几乎总是能够胜过它们。人们可能会注意到FedMA的曲线是阶梯状的，这是因为FedMA在每一轮通信中只更新一层的参数，这需要11轮才能更新VGG11中的所有11层。如果仅看 FedMA 曲线的提升点，FedMA 的表现超过 FedAvg。 OTFusion 似乎不适合联邦学习，因为它比 FedAvg 还差。

**该实验证明了我们的方法 GAMF 的泛化能力，因为它可以在紧凑模型集成和联邦学习实验中工作。**

#### Experiments on CIFAR-100
图 5c 和 5d 分别显示了具有 5 个客户端和 10 个客户端的 CIFAR-100 上的结果。在 5 个客户端设置中，我们可以看到 GAMF 在整个训练过程中可以优于所有基线。**我们承认最终的准确率没有我们预期的那么好**，我们认为这是因为 VGG11 很难在 CIFAR-100 上进行联邦学习。与FedMA相比，GAMF可以将模型融合的性能提升约3%。

在 10 个客户端的实验中，我们将通信轮数从 55 增加到 100。如图 5d 所示，Moon 的最终性能略高于我们的 GAMF，但仍在可接受的范围内。 GAMF在训练过程的中期表现超过了Moon，这意味着GAMF的收敛速度是比较显着的。尽管在 Moon 之下，GAMF 的表现优于其他基线 FedMA、FedAvg 和 OTFusion。遗憾的是，OTFusion 未能在 CIFAR-100 上收敛。

#### Trade off between time and accuracy
与 OTFusion 相比，我们提出的 GAMF 考虑了二阶相似性。二阶相似性可以更好地对齐通道，但有一个明显的缺点是额外时间的成本。因此，我们将OTFusion和GAMF的时间消耗记录在表4中，以直观地看到GAMF由于二阶相似性而使用的额外时间。**我们可以看到，GAMF的时间消耗仅比OTFusion高50%左右。我们认为这种额外时间是可以接受的，因为与 OTFusion 相比，GAMF 的性能提升超过 50%。**
![500](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230704110707.png)

#### Combination of Moon and GAMF
**Moon 是最先进的 FL 方法**，专注于改进每个客户端的本地训练，这与 GAMF 专注于模型聚合不同。我们发现 Moon 的性能在某些设置下与我们的 GAMF 类似。这表明 Moon 和 GAMF 对 FL 都有用。 Moon与GAMF由于侧重点不同，**是互补而非竞争关系**。因此，我们进行了将 GAMF 与 Moon 结合的额外实验。我们展示了具有相同实验设置的组合方法的性能。结果如表3和图5所示。事实证明，与单独的GAMF和Moon相比，GAMF和Moon的组合确实达到了更好的性能。它还表明模型融合是 FL 的一个重要因素，**因为 GAMF 提升的性能接近甚至优于 Moon 提升的性能。**

#### Experiments on larger scale
在上述实验中，我们使用 5/10 客户端测试了 GAMF 在 CIFAR-10、CIFAR-100 和 Tiny-Imagenet 上的性能。在这里，我们与更多的客户进行了额外的实验，以了解 GAMF 的可扩展性。如表5所示，我们在CIFAR-10上将FedAvg、Moon、GAMF和GAMF+Moon与10/16/20个客户端进行比较，这是联邦学习论文实验中常见的客户端数量。**我们可以看到，我们提出的 GAMF 在更多客户端场景下仍然可以优于基线。这些实验证明了我们提出的 GAMF 的可扩展性。**

![500](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230704111003.png)

## Q7 用于定量评估的数据集和baselines是什么？

## Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

## Q9 这篇论文到底有什么贡献？
1. 与之前的工作不同，在不访问训练数据的情况下使用线性分配问题（LAP）来解决模型融合的新兴范例，我们设法通过图 1 https://www.kaggle.com/c/tiny-imageNet 与边缘信息匹配来解决它。据我们所知，我们是第一个解决 LAP 之外的模型融合问题的人。
2. 我们在GM框架下提出了一种可扩展的分级分配方法GAMF，设计了一种迭代内存高效的切片和扫描过程，这是基于我们对问题内在稀疏性特征的关键观察，因为通道匹配应限制在同一层。
3. 我们考虑多模型融合中的循环一致性属性来改进所提出的GAMF，这在之前的模型融合文献中没有研究过。
4. 实验证明了 GAMF 在紧凑模型集成和联邦学习中的有效性。我们的源代码位于：https://github.com/Thinklab-SJTU/GAMF。


在本文中，我们设法将模型融合表述为图匹配问题。为了可扩展性和一致性，我们提出了一种名为 GAMF 的基于分级分配的方法，用于融合两个或多个模型，该方法捕获并适应融合问题固有的稀疏性。在紧凑模型集成和 FL 的任务上，GAMF 优于同行方法。诚然，即使我们设计了高效的算法，由于图匹配的复杂度较高，GAMF 与 OTFusion 相比仍具有额外的时间复杂度。然而，更好的融合模型可能值得这样的开销，并且在 FL 中，瓶颈通常是通信成本，而我们的方法可以显着降低通信成本。未来我们将探索不同规模模型和更多客户端的模型融合，以提高GAMF的可扩展性。**此外，我们计划通过多图匹配从传统方法（Jiang et al., 2020）到基于学习的方法（Rolı́nek et al., 2020）来改进多模型融合，这是组合优化交叉领域的一个新兴领域和机器学习。**
## Q10 下一步呢？有什么工作可以继续深入？

### Ideas
- **具有异构数据分区的 4 个模型的设置对于模型融合方法来说似乎太难了，因为所有方法都不能很好地工作。此设置中的微调更像是从头开始重新训练网络。**==（一个可以改进的工作点！！）==
- FL相关的baseline中经常会拉上FedAvg
- **Moon 是最先进的 FL 方法**，可以研究一下这篇论文看看能不能改进
- **我们可以看到，GAMF的时间消耗仅比OTFusion高50%左右。我们认为这种额外时间是可以接受的，因为与 OTFusion 相比，GAMF 的性能提升超过 50%。**
  或许可以考虑改进GAMF的耗时
- 此外，我们计划通过多图匹配从传统方法（Jiang et al., 2020）到基于学习的方法（Rolı́nek et al., 2020）来改进多模型融合，这是组合优化交叉领域的一个新兴领域和机器学习。
  （也可以试试组合优化！）
## New Concepts


%% Import Date: 2023-06-25T17:25:33.060+08:00 %%
