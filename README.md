## Graph Harmony: Denoising and Nuclear-Norm Wasserstein Adaptation for Enhanced Domain Transfer in Graph-Structured Data
The official implementation of "Graph Harmony: Denoising and Nuclear-Norm Wasserstein Adaptation for Enhanced Domain Transfer in Graph-Structured Data", TMLR 2024.

### Requirements
* python==3.9
* numpy==1.23.1
* scikit_learn==1.1.2
* torch==1.12.1
* torchvision==0.13.1

### Datasets
The Ego-network dataset with 4 domains: OAG, Digg, Twitter, and Weibo
* [Raw Ego-network dataset](https://github.com/xptree/DeepInf)
* [Processed Ego-network dataset](https://drive.google.com/drive/folders/1Kvd46c1TtbuL3-svUC2fkjPnwMOXy8wH?usp=sharing)

The IMDB&Reddit dataset with 2 domains: IMDB-Binary and Reddit-Binary
* [Raw IMDB-Binary dataset](https://networkrepository.com/IMDB-BINARY.zip)
* [Raw Reddit-Binary dataset](https://networkrepository.com/REDDIT-BINARY.zip)
* [Processed IMDB-Binary dataset](https://drive.google.com/drive/folders/1Kvd46c1TtbuL3-svUC2fkjPnwMOXy8wH?usp=sharing)
* [Processed Reddit-Binary dataset](https://drive.google.com/drive/folders/1Kvd46c1TtbuL3-svUC2fkjPnwMOXy8wH?usp=sharing)

Download the datasets and put them into a folder called data_folder.
### Baselines
[DANN](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf), [MDD](https://arxiv.org/abs/1904.05801), [DIVA](https://arxiv.org/abs/1905.10427), [BIWAA](https://ieeexplore.ieee.org/document/10030755), [SDAT](https://arxiv.org/abs/2206.08213), [ToAlign](https://arxiv.org/abs/2106.10812)

### Arguments
-backbone: Feature Extractor <br>
-r: The random seed = r + 27<br>
-data_path: The folder contains data files<br>
-src_data: The source dataset<br>
-tar_data: The target dataset<br>

<div style="page-break-after: always; visibility: hidden"> 
\pagebreak 
</div>


### Training with Our Method
- Train on Ego-network dataset
```bash=
python run_ours.py \
--backbone gat \
-r 0 \
--data_path data_folder/infodata/data/ \ 
--src_data digg --tar_data oag --device cuda 
```

- Train on IMDB&Reddit dataset
```bash=
python run_ours_ir.py \
--backbone gat \
-r 0 \
--data_path data_folder/ \
--src_data REDDIT-BINARY --tar_data IMDB-BINARY --device cuda
```


### Reference
This repo is developed based on the codes provided by [GraphDA](https://github.com/rynewu224/GraphDA)

