# Fully Convolutional Networks for Semantic Segmentation

### Sample Output
![alt text](./images/title.png "Sample output")

## Dataset
 - Traning Data:  **BSD**   [[link](http://home.bharathh.info/pubs/codes/SBD/download.html)]
 - Testing Data:  **VOC 2012**   [[link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#voc2012vs2011)] <br><br>
 Both BSD and VOC 2012 Dataset contains 21 categories of object.
 [[Category list](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)] <br>

## Segmentation Class
[[Class list](seg_class)]
 
 
## Network
 - **FCN32**
   - **VGG16**   [[VGG16 for tensorflow](https://github.com/machrisaa/tensorflow-vgg)] <br>
     (_Transfer Learning_) Using pretrained VGG19 weightings except the last prediction layer, which
     was replaced and trained with a new upscore layer to perform upsampling. 
