---
citekey: 2022LayerwisedModelAggregationPersonalizedFederated
Type: journalArticle
Title: 'Layer-wised Model Aggregation for Personalized Federated Learning (pFedLA)'
Author: 'Xiaosong Ma, Jie Zhang, Song Guo, Wenchao Xu'
Journal: '2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)'
Date: 2022-06-01
DOI: 10.1109/CVPR52688.2022.00985 
---

# Layer-wised Model Aggregation for Personalized Federated Learning (pFedLA)

## One-sentence summary
这篇文章比较创新的一点就是把FL从按本地模型聚合，细化到了按层聚合，按层聚合的过程中针对每层权重的问题，引入了超网络的思想来给每个客户端进行参数调节，很像是元学习的思想。特别是第二个框架HeurpFedLA，虽然并没有降低多少通信效率，但是的确一定程度上做到了pFL问题的“各客户端的特殊性”和“全局模型的普适性”之间的权衡。111333555
## Metadata

>[!info] Info
>- **Title**: Layer-wised Model Aggregation for Personalized Federated Learning (pFedLA) 
>- **Author**: Xiaosong Ma, Jie Zhang, Song Guo, Wenchao Xu
>- **Year**: 2022-06-01
>- **Bibliography**: Ma, X., Zhang, J., Guo, S., & Xu, W. (2022). Layer-wised Model Aggregation for Personalized Federated Learning (pFedLA). _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 10082–10091. [https://doi.org/10.1109/CVPR52688.2022.00985](https://doi.org/10.1109/CVPR52688.2022.00985)

>[!abstract] Abstract
> Personalized Federated Learning (pFL) not only can capture the common priors from broad range of distributed data, but also support customized models for heterogeneous clients. Researches over the past few years have applied the weighted aggregation manner to produce personalized models, where the weights are determined by calibrating the distance of the entire model parameters or loss values, and have yet to consider the layer-level impacts to the aggregation process, leading to lagged model convergence and inadequate personalization over non-IID datasets. In this paper, we propose a novel pFL training framework dubbed Layer-wised Personalized Federated learning (pFedLA) that can discern the importance of each layer from different clients, and thus is able to optimize the personalized model aggregation for clients with heterogeneous data. Specifically, we employ a dedicated hyper-network per client on the server side, which is trained to identify the mutual contribution factors at layer granularity. Meanwhile, a parameterized mechanism is introduced to update the layer-wised aggregation weights to progressively exploit the inter-user similarity and realize accurate model personalization. Extensive experiments are conducted over different models and learning tasks, and we show that the proposed methods achieve significantly higher performance than state-of-the-art pFL methods.

> [!example] Files and Links 

>- **DOI**: 10.1109/CVPR52688.2022.00985
>- **Url**: https://ieeexplore.ieee.org/document/9880164/
>- **PDF**: [2022_Ma et al_Layer-wised Model Aggregation for Personalized Federated Learning.pdf](file://C:\Users\可可\OneDrive%20-%20sjtu.edu.cn\Zotero\storage\20222022%20IEEECVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR)\2022_Ma%20et%20al_Layer-wised%20Model%20Aggregation%20for%20Personalized%20Federated%20Learning.pdf)
>- **Zotero**: [2022_Ma et al_Layer-wised Model Aggregation for Personalized Federated Learning.pdf](zotero://select/library/items/4JPMQ6LS)

> [!tip] Zotero Tags
>- **Keywords**: #unread, #CCF-A

## Annotations


---
# Obsidian Notes


## 辅助资料
[联邦学习论文阅读：Layer-wised Model Aggregation for Personalized Federated Learning](http://jay-codeman.life/index.php/archives/656/)

## What can we learn


## Q1 论文试图解决什么问题？

## Q2 这是否是一个新的问题？

## Q3 这篇文章要验证一个什么科学假设？

## Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

## Q5 论文中提到的解决方案之关键是什么？
![](https://cdn.jsdelivr.net/gh/LonelyMoonDesert/BlogImgBed2@main/img/20230419205135.png)
## Q6 论文中的实验是如何设计的？

## Q7 用于定量评估的数据集和baselines是什么？

## Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

## Q9 这篇论文到底有什么贡献？

## Q10 下一步呢？有什么工作可以继续深入？


## New Concepts
### 超网络Hypernetworks
[什么是超网络？](https://zhuanlan.zhihu.com/p/34038294)
> 超网络**仅仅是一个小型网络，该网络为一个大得多的网络生成权重**，比如为一个深度残差网络生成权重，高效地设定残差网络每层的权重。 我们可以使用超网络来探索模型的表达力和捆绑深度卷积网络权重的程度之间的权衡。 这有些像压缩图像时可以调整压缩率，超网络不过是把图像换成深度卷积网络的权重。


%% Import Date: 2023-04-19T20:09:20.322+08:00 %%
