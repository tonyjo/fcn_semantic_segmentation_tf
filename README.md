# Fully Convolutional Networks for Semantic Segmentation

### Sample Output
![alt text](./images/title.png "Sample output")

## Dataset
 - Traning Data:  **BSD**   [[link](http://home.bharathh.info/pubs/codes/SBD/download.html)]
 - Testing Data:  **VOC 2012**   [[link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#voc2012vs2011)] <br><br>
 Both BSD and VOC 2012 Dataset contains 21 categories of object.
 [[Category list](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)] <br>

#### Segmentation Class
[[Class list](seg_class)]
 
## Prerequisites
1. Tensorflow Version --v1.4 with CUDA > 8.0
2. Numpy --v1.15
3. OpenCV --v4.0
4. Matplotlib --v2.0

### Training
Once the dataset is obtained, the model properties can be configured in config_files directory:
```
config_files
│   config_fcn16s_train.yaml
│   config_fcn32s_train.yaml    
```
and then run the following command:
```bash
./run0.sh config_files/config_fcn32s_train.yaml
```
or
```bash
./run0.sh config_files/config_fcn16s_train.yaml
```

### Results
Coming Soon.

### To-Do
FCN-8s model

#### Paper citation
```
@article{ShelhamerLD17,
  author    = {Evan Shelhamer and
               Jonathan Long and
               Trevor Darrell},
  title     = {Fully Convolutional Networks for Semantic Segmentation},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume    = {39},
  number    = {4},
  pages     = {640--651},
  year      = {2017}
}
```
