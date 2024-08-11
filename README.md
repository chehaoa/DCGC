# DCGC
An official source code for paper Dual-Center Graph Clustering Based on Neighbor Distribution

-------------

### Overview

<p align = "justify"> 
Graph clustering is crucial for unraveling intricate data structures, yet it presents significant challenges owing to its unsupervised nature. Recently, target-driven clustering techniques have yielded impressive results, with contrastive learning methods leveraging pseudo-labels garnering considerable attention. Nonetheless, existing approaches rely on label distribution for network learning supervision, limiting them to single-center clustering optimization, resulting in incomplete and less robust guidance. In this work, we introduce two key enhancements to current methods based on neighbor distribution properties and propose a novel Dual-Center Graph Clustering (DCGC) approach. Our method incorporates neighbor distribution representation learning and a dual-clustering-center optimization module. By utilizing neighbor distribution as a supervisory signal, we bolster network representation learning, demonstrating enhanced robustness compared to relying solely on labels. Furthermore, we integrate neighbor distribution centers with embedding centers, enforcing dual-center constraints for network optimization. Through extensive experiments and analyses, we showcase the superior performance and effectiveness of our proposed method. 
<div  align="center">    
    <img src="./assets/HSAN_model.png" width=80%/>
</div>


<div  align="center">    
      Figure 1: Illustration of our proposed dual-center graph clustering approach (DCGC).
</div>

### Requirements

The proposed HSAN is implemented with python 3.7 on an NVIDIA A100 tensor core GPU. 

Python package information is summarized in **requirements.txt**:

- torch==1.7.1
- tqdm==4.59.0
- numpy==1.19.2
- munkres==1.1.4
- scikit_learn==1.2.0



### Quick Start

- Step1: use the **cora.zip** file or download other datasets from  [Awesome Deep Graph Clustering/Benchmark Datasets](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering#datasets-details) 

- Step2: unzip the dataset into the **./dataset** folder

- Step3: "mkdir pretrain" to create **./pretrain** folder

- Step4: run

  ```
  python train.py
  ```

  the clustering results will be recorded in the **./results.csv** file
