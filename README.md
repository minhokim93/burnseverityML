# Explaining Wildfire Burn Severity Factors Using Machine Learning
Authors: [Minho Kim](https://minho.me), Weixin Li, Fengzhe Lin

Background
---------------------
With the exacerbation of climate change effects, communities in the Wildland Urban Interface (WUI) are at risk of being devastated by higher severity and more frequent wildfires over time. However, WUI landscapes are heterogeneous and mixed with highly complex features. In response, there is an urgent need to understand the wildfire-causing factors towards the development of effective and sustainable wildfire mitigation. However, there is a disconnect between our understanding of landscape patterns and wildfire burns, especially at a high spatial resolution needed to resolve the highly heterogeneous and complex WUI landscapes. We aim to develop a data-driven, machine learning approach to assess wildfire burn severity with high resolution remote sensing data and generate an understanding of burn severity with different landscape-related variables.


All experiments were trained from scratch and were performed using an Intel Core i7-6700 CPU at 3.40 GHz and an NVIDIA GeForce RTX 2070 Super Graphics Processor Unit (GPU) with 8 GB of memory. Python 3.7.9 was used with Tensorflow 2.3.0. For training hyperparameters, an early stop of 15 epochs, a learning rate of 0.002, and a decay factor of 0.004 were used. The adaptive moment estimation (adam) optimizer was chosen to minimize the cross-entropy loss function. Filter weights were initialized using “He normal” initialization.

Requirements
---------------------
- python=3.7.9
- scikit-learn
- matplotlib
- pandas

Usage
---------------------
1. Install a new conda environment
```
$ conda env create --name howsevere --file environment.yml
```
2. Activate the new environment and navigate to the "src" folder
```
$ conda activate howsevere
$ cd src
```

**:fire: State-of-the-art Explainable AI studies on Burn Severity**
---------------------
1.


Citation
---------------------
**Please cite the journal paper if this code is useful and helpful for your research.**

