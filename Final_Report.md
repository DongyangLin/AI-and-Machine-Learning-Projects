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