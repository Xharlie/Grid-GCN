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
  We refer to pointnet  https://github.com/charlesq34/pointnet/blob/master/provider.py
  ```
  cd data/
  wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
  unzip modelnet40_ply_hdf5_2048.zip
  unzip it and put it inside data/
  ```
  * #### ModelNet10
  please refer to pointnet++'s github
  ```
  download  modelnet40_normal_resampled from https://github.com/charlesq34/pointnet2
  take the modelnet10_train.txt, modelnet10_test.txt and extract from modelnet40_ply_hdf5_2048 to create a modelnet10_ply_hdf5_2048
  or use modelnet40_normal_resampled directly, but configuration file configs_10.yaml new: True -> False
  ```
  
* ### Segmentation/ScanNet
  Please refer to pointnet++ for downloading ScanNet use link: 
  ```
  # in data/
  wget https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip
  unzip scannet_data_pointnet2.zip
  mv data scannet


## Training
* ### Classification

  * #### ModelNet40
  ```
  cd classification
  nohup python -u train/train_ggcn_mdl40.py &> mdl40.log & 
  
  ```
  * #### ModelNet10
  please refer to pointnet++
  ```
  nohup python -u train/train_ggcn_mdl10.py &> mdl10.log &
  
  ```
  
* ### Segmentation 
  * #### ScanNet
  Please refer to pointnet++ for downloading ScanNet use link: 
  ```
  cd segmentation
 
  nohup python -u train/train_ggcn_scannet.py &> log  &
  ```
