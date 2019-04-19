# PointNet.pytorch

This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch.\
The model is in `pointnet/model.py`.

Code has been tested on the following machine:

| HW        | Specs                                    |
|-----------|------------------------------------------|
| OS (type) | Ubuntu 18.04.2 LTS (64-bit)              |
| Memory    | 62,8 GiB                                 |
| Processor | Intel® Core™ i7-5930K CPU @ 3.50GHz × 12 |
| Graphics  | GeForce GTX 1080/PCIe/SSE2               |

| SW            | Version |
|---------------|---------|
| Python        | 3.6.7   |
| Nvidia CUDA   | 10.0    |
| Nvidia driver | 410.48  |

| Python packages | Version     |
|-----------------|-------------|
| torch (PyTorch) | 1.0.1.post2 |
| torchvision     | 0.2.1       |


## Setup

Download repository and install dependencies:

```bash
git clone https://github.com/mcarletti/pointnet.pytorch.git
cd pointnet.pytorch
pip3 install -e .
```

Download dataset (subset of ShapeNet):

```bash
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
```

[ optional ] Download and build visualization tool
```bash
cd script
bash build.sh # build C++ code for visualization
bash download.sh # download dataset
```

## Training

Training classification and segmentation models.\
**BUG** Batch size must be at least 2.

```bash
cd utils
python3 train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python3 train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

Use `--feature_transform` to use feature transform.

Usage example:
```bash
# original pointnet training procedure
python3 train_classification.py --dataset data/shapenetcore_partanno_segmentation_benchmark_v0 --nepoch 10 --batchSize 32 --dataset_type shapenet

# example of custom usage
python3 train_classification.py --dataset /media/Data/userdata/mcarletti/pointnet/pts/pointclouds/ --nepoch 10 --batchSize 32 --dataset_type plyweb
```

In order to debug the code using `import pdb; pdb.set_trace()`, you must add the argument `--workers 0` to avoid using parallel computation.

# Performance

## Classification performance

On ModelNet40:

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | 89.2 | 
| this implementation(w/o feature transform) | 86.4 | 
| this implementation(w/ feature transform) | 86.8 | 

On [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html)

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | N/A | 
| this implementation(w/o feature transform) | 98.1 | 
| this implementation(w/ feature transform) | 96.8 | 

## Segmentation performance

Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Original implementation |  83.4 | 78.7 | 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6| 
| this implementation(w/o feature transform) | 73.5 | 71.3 | 64.3 | 61.1 | 87.2 | 69.5 | 86.1|81.6| 77.4|92.7|41.3|86.5|78.2|41.2|61.0|81.1|
| this implementation(w/ feature transform) |  |  |  |  | 87.9 |  | | | | | | | | | |81.4|

Note that this implementation trains each class separately, so classes with fewer data will have slightly lower performance than reference implementation.

Sample segmentation result:
![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)

# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)
