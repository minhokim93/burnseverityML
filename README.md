# Explaining Wildfire Burn Severity Factors Using Machine Learning
Authors: [Minho Kim](https://minho.me), Weixin Li, Fengzhe Lin


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

