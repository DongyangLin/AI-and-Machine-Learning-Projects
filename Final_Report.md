#  Exploring Clustering and Dimensionality Reduction in Data Analysis

## Abstract:

This project investigates the application of clustering techniques and dimensionality reduction methods in data analysis. Implementing K-means and Soft K-means algorithms in Python using Numpy, we evaluate their performance on a diverse dataset with seven input features and three distinct labels. The project explores clustering effectiveness without label information and introduces non-local split-and-merge moves to enhance algorithmic accuracy. Additionally, dimensionality reduction techniques, PCA and Linear Autoencoder, are implemented to showcase their versatility.

## Keywords:

 Clustering, K-means, Soft K-means, Dimensionality Reduction, PCA, Linear Autoencoder, Data Analysis, Non-local Split-and-Merge Moves.

## Introduction:

This project aims to construct machine learning models for two tasks: clustering and dimensionality reduction, investigating their performance.

Firstly, for the clustering task, KMeans algorithm models and Soft KMeans algorithm models are constructed using Python and Numpy. The seed dataset is employed to evaluate the clustering performance of both models. The models' accuracy is horizontally compared by assessing the clustering results against the true labels. Additionally, optimization techniques are applied to enhance the models, such as refining the initialization method for cluster centers to achieve better and more efficient clustering performance.

Secondly, non-local split-and-merge moves are incorporated into the basic KMeans and Soft KMeans algorithms. Parameters such as split threshold, merge threshold, and initial cluster quantity are adjusted to enhance the models' automatic optimization capabilities.

For the dimensionality reduction task, PCA (Principal Component Analysis) models and Linear Autoencoder models are constructed using Python and Numpy. The models' performance is evaluated by comparing the dimensionality reduction and reconstruction results of RGB color images and grayscale images. The similarity between the reconstructed images and the original images serves as an indicator of the model's performance. Additionally, due to the limited capacity of Python and Numpy in handling data, the Linear Autoencoder is tested using images from the Fashion MNIST dataset.

In terms of optimizing the Linear Autoencoder model, adjustments are made to the dimensions (number of neurons) of each hidden layer. This aims to achieve dimensionality reduction results with the smallest possible dimensions, retaining the minimal components while recovering images as close as possible to the original ones.

This project seeks to provide comprehensive insights into the performance and optimization of machine learning models for clustering and dimensionality reduction tasks.

## Problem Formulation:

The project addresses several key challenges inherent in data analysis:

1. **Clustering Analysis:** Evaluate the performance of K-means and Soft K-means clustering algorithms on datasets with diverse characteristics. Assess their effectiveness in uncovering patterns without using label information.

2. **Impact of Cluster Number:** Investigate the consequences of setting K=10 in both K-means and Soft K-means. Analyze the algorithms' ability to handle an increased number of clusters and differentiate finer distinctions within the dataset.

3. **Algorithm Enhancement:** Modify clustering algorithms by incorporating non-local split-and-merge moves. Observe the impact of these enhancements on clustering accuracy, particularly with a higher number of clusters.

4. **Dimensionality Reduction:** Implement PCA and Linear Autoencoder to extract essential features and reconstruct original data. Showcase the potential of these methods in simplifying data representation across diverse datasets.

The project aims to provide insights into the strengths and limitations of clustering and dimensionality reduction techniques, offering valuable implications for a broad range of data analysis scenarios.

## Method and Algorithms

### Methods:

本项目通过模型构建、数据集测试，控制变量和模型横向对比的方法，构建出合适的聚类模型和数据降维模型。并且通过多种数据集的测试优化模型的表现。

其中KMeans和SoftKMeans聚类模型通过wheat seed dataset和彩色图片的聚类进行对比。在此基础上引入non-local split-and-merge moves优化模型，并且通过对wheat seed dataset设置更高的聚类数量进一步判断模型表现。

对于数据降维模型，通过谷歌的Fashion MNIST（包含28*28的黑白图片）数据集以及CIFAR-10数据集（包含32\*32的彩色图片）训练线性自动编码器并且对比原图分析模型表现。另外，用PCA模型对这两个数据集进行特征提取，对比PCA的重构图片和线性自动编码器的重构图片，分析两个不同模型的表现。

### Algorithms:

#### KMeans:

KMeans算法通过初始化中心点、中心点移动以及聚类更新将数据划分为不同的类别达到聚类效果。KMeans的损失函数可以定义为每个聚类数据点到聚类中心的欧氏距离之和：
$$
\min_{\{\mathbf m\},\{\mathbf r\}}J(\{\mathbf m\},\{\mathbf r\})=\min_{\{\mathbf m\},\{\mathbf r\}}\sum_{n=1}^{N}\sum_{k=1}^{K}r_k^{(n)}||\mathbf m_k-\mathbf x^{(n)}||^2
$$
其中，$\mathbf x \in \mathbb{R}^{n \times d}$为样本n个d维样本输入。$\mathbf m \in \mathbb{R}^{k \times d}$为聚类中心，k为聚类数量。$r_k^{(n)}\in \{0,1\}$为样本分类的标识符，采用独热编码，1代表该数据点属于第k个聚类中心。

通过算法迭代使损失函数达到最小值，具体步骤如下：

1. 初始化聚类中心：对于某一数据集，设置k个聚类中心，通过随机数生成等方法初始化聚类中心。

2. 聚类：通过判断每个数据点到聚类中心的距离，挑选离其最近的聚类中心作为那一点所属的类别。
   $$
   \hat{k}^{(n)}=\arg \min_k||\mathbf m_k-\mathbf x^{(n)}||^2
   $$
   且
   $$
   r_k^{(n)}=1\leftrightarrow \hat{k}^{(n)}=k
   $$
   
3. 聚类中心移动：聚类步骤完成后，将聚类中心移动到对应聚类数据点的中心。
   $$
   \mathbf m_k=\frac{\sum_n r_k^{(n)} \mathbf x^{(n)}}{\sum_n r_k^{(n)}}
   $$

4. 更新：聚类中心移动后，重复步骤2和步骤3，直到聚类中心不再变更或达到最大迭代次数。

#### SoftKMeans:

为了避免过于绝对的聚类方式，引入Soft KMeans让模型在聚类过程中获取更多数据信息。在每次更新聚类时，调整聚类标识符$r_k^{(n)}$为
$$
r_k^{(n)}=\frac{\exp [-\beta ||\mathbf m_k-\mathbf x^{(n)}||^2]}{\sum_j\exp [-\beta ||\mathbf m_j-\mathbf x^{(n)}||^2]}
$$
聚类中心更新过程保持不变。

#### KMeans++:

为了尽可能避免聚类中心重合以及陷入局部最优解，引入KMeans++方法对KMeans和SoftKMeans的聚类中心进行初始化。

K-Means++ 的初始中心选择过程如下：

1. 从数据点中随机选择第一个中心。
2. 对于每个数据点，计算其与已选择中心的最短距离（即与最近的中心的距离）。
3. 按照这些距离的概率分布，选择下一个中心。距离越远的点，被选择为下一个中心的概率越大。
4. 重复步骤 2 和步骤 3，选择剩余的中心，直到选择了 k 个初始中心。

这样，K-Means++ 确保了初始中心点的广泛分布，使得算法更有可能找到全局最优解，减少了收敛到局部最优解的风险。

#### PCA:

主成分分析算法是通过提取输入数据主成分达到降维目的的算法。此算法可以有效压缩数据集，并且通过压缩后的数据集恢复出与原有数据集相似的数据。

##### 数据压缩：

对于任意输入数据$\mathbf x \in \mathbb {R}^{d}$，可通过压缩重构近似表达为
$$
\mathbf x \approx  U\mathbf z + \mathbf a
$$
其中，$U\in \mathbb {R}^{d \times m}$为投影矩阵，$\mathbf z \in \mathbb {R}^{m}$为压缩后的数据，$\mathbf a \in \mathbb {R}^d$为中心偏移量补偿。

为了寻找投影矩阵$U$，构建协方差矩阵$C\in \mathbb {R}^{d \times d}$
$$
C=\frac{1}{N}\sum_{n=1}^{N}(\mathbf x^{(n)}-\mathbf {\bar x})(\mathbf x^{(n)}-\mathbf {\bar x})^T
$$
其中$M$个最大的特征值即为所寻找的主成分。由此，可以将协方差矩阵写为
$$
C=U\Sigma U^T \approx U_{1:M}\Sigma_{1:M}U_{1:M}^T
$$
其中$U$为所需的投影矩阵，由$M$个最大特征值对应的特征向量组成。且矩阵$U$单位正交
$$
U^TU=UU^T=1
$$
$\Sigma $为协方差矩阵特征值构成的对角阵。

得到投影矩阵$U$后，将原始数据$\mathbf x$投影即可得到压缩后的数据$\mathbf z$。
$$
\mathbf z=U^T \mathbf x
$$
PCA满足最大方差（绿色点），和最小误差的性质（红色点），即：在$\mathbf u$方向投影得到的$\widetilde {\mathbf x}_n$之间间隔距离最大，且投影后的数据$\widetilde {\mathbf x}_n$和原始数据$\mathbf x_n$距离最小。

<img src="C:\Users\zikun\AppData\Roaming\Typora\typora-user-images\image-20231220120301435.png" alt="image-20231220120301435" style="zoom:40%;" />

##### 数据重构：

完成PCA的数据降维后，可以根据降维过程将数据重构，恢复出与原来近似的数据$\widetilde {\mathbf x}_n$。重构过程需要最小化重构误差函数
$$
J(\mathbf u,\mathbf z,\mathbf b)=\sum_n||\mathbf x^{(n)}-\widetilde {\mathbf x}^{(n)}||^2
$$
其中
$$
\widetilde {\mathbf x}^{(n)}=\sum_{j=1}^{M}z_j^{(n)}\mathbf u_j + \sum_{j=M+1}^D b_j\mathbf u_j
$$
符合最小化重构函数的$z_j^{(n)}$和$b_j$为
$$
z_j^{(n)}=\mathbf u_j^T\mathbf x^{(n)} \\
b_j = \mathbf{\bar x}^T\mathbf u_j
$$
得到$z_j^{(n)}$的过程即为投影的逆过程，$b_j$为中心偏移量的补偿。

#### 线性自动编码器：

线性自动编码器与多层感知机（MLP）类似，将输入数据作为自己的输出数据，通过神经网络对数据特征进行学习和提取，能够将数据压缩到更低维度，且重构出与原数据几乎相同的数据。

<img src="C:\Users\zikun\AppData\Roaming\Typora\typora-user-images\image-20231220132515597.png" alt="image-20231220132515597" style="zoom:50%;" />

线性自动编码器可以分为两部分，前半部分为编码器，可以将数据降至更低维度，后半部分为解码器，可以将降维后的数据进行重构。训练完成后，中间最低维度的隐藏层数据即为所需要的降维后数据。

线性自动编码器与MLP模型的区别在于输出层无需激活函数，即线性输出
$$
\mathbf y=\mathbf w_0+\mathbf w^T \mathbf h
$$

## Experiment results and analysis

### KMeans and Soft KMeans

通过含有210个数据点，每个数据点7个特征的wheat seed dataset对KMeans和Soft KMeans进行表现评估。该数据集总共有三个标签，因此设置聚类个数K=3。对比数据集的真实标签和两个模型的聚类结果，KMeans模型和Soft KMeans模型的聚类准确率均为71.67%，二者表现相近。这可能是由于引入KMeans++方法后，两者的聚类性能都得到显著提升，可以避免陷入局部最优解，从而拉近了两者的聚类效果差距。

