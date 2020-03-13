# GGCN implemented by MXNET 1.5.0

make sure you have gcc version suggested by MXNET 1.5.0

## Install Our CUDA modules to MXNET Libary:
```
cd gridifyop
vim Makefile  # then change mx_home to your mxnet-apache directory, and adjust nvcc command according to your gpu model and cuda version. here we use compute power 61 and 75 for 1080 ti and 2080 ti. save the change
make
cd ..
```

## Data Preparation

* ### Classification

  * #### ModelNet40
  ```
  Preprocessed Modelnet40 dataset can be downloaded https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip. 
  Rename the uncompressed data folder as data/modelnet40_normal_resampled.
  ```
  
* ### Segmentation 
  * #### ScanNet
  Please refer to pointnet++ for downloading ScanNet use link: 
  ```
  cd data/
  wget https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip
  unzip scannet_data_pointnet2.zip
  mv data scannet


## Training
* ### Classification

  * #### ModelNet40
  ```
  cd classification
  nohup python -u train/train_ggcn_mdl40.py &> log & 
  
  ```
  
* ### Segmentation 
  * #### ScanNet
  Please refer to pointnet++ for downloading ScanNet use link: 
  ```
  cd segmentation
 
  nohup python -u train/train_ggcn_scannet.py &> log  &
  ```
