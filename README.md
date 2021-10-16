    Pytoch implementation of rankCNN for brain disease prognosis

    The code was written by Hezhe Qiao and Dr. Lin Chen, Department of Radiology at Chongqing Institute of Green\
     and Intelligent Technology, Chinese Academy of Sciences. 

1. Introduction
We proposed a ranking convolutional neural network (rankCNN) to address
the prediction of MMSE through muti-classification. Specifically, we use a 3D convolutional neural
network with sharing weights to extract the feature from MRI, followed by multiple sub-networks
which transform the cognitive regression into a series of simpler binary classification. In addition, we
further use a ranking layer measure the ranking information between samples to strengthen the ability
of the classification by extracting more discriminative features.
  
2. Prerequisites
Linux python 3.7 Pytorch version 1.2.0 NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 10.0.61
 

3.Dataset
The dataset used in this study was obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) that is avaiable at http://adni.loni.usc.edu/ .


4.Pre-process
All MRIs were prepocessed by a standard pipeline in CAT12 toolbox which is avaiable at http://dbm,neuro.uni-jena.de/cat/.

5.Note
Please cite our paper if you use this code in your own work.

