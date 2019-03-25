CheXpert
==============================

An entry for the CheXpert competition from the Stanford ML group. The goal is to create a multi-label classifier for chest X-ray image that outputs probabilities of 14 different observations (including 12 pathologies, "No Finding", and "Support Devices").

I will set uncertainty u = 1 (as described in their paper) for the first trial run.

The CheXpert paper uses an Adam optimizer withh default beta and constant learning rate = 1e-4, uses batch normalization with a batch size of 16, 3 epochs, and saved checkpoint every 4800 iterations. 

The original DenseNet paper used Nesterov momentum without dampening, stochastic gradient descent, learning rate decay and weight decay. They use Batch normalization and dropout = 0.2. With their ImageNet model they used mini-batch gradient descent without dropout.

When used together with batch normalization, weight decay and learning rate decay are no more independent. 

One modification worth making to the Stanford ML group's implementation is to use the AdamWR algorithm, which has a corrected weight decay with a normalized batch norm, and uses cosine annealing that resets on every batch to attempt to improve the speed of convergence.

### Datasets:

[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

[MIMIC-CXR](https://physionet.org/physiobank/database/mimiccxr/)

### References:

[CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031)

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

[Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf)
