# Semantic Segmentation

[//]: # (Image References)
[image0]: ./images/architecture.png
[image1]: ./images/result/uu_000005_e1.png
[image2]: ./images/result/uu_000005_e20.png
[image3]: ./images/result/uu_000005_e35.png


## Overview
The goal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN).


## Architecture
![Architecture][image0]

This implementation is based on the 2015 paper [Fully Convolutional Networks](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) for Semantic Segmentation from UC Berkeley. 

In addition to the VGG, I integrated skip layers, 1x1 convolution and upsampling.

The model has been trained on Amazon GPU instance. (https://aws.amazon.com/?nc2=h_lg)

The main section of the code of the architecture can be seen below:

```c++
# predict1
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Deconv1
    upsampling1 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, strides=(2, 2), padding='same',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # predict2
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    skip1 = tf.add(conv_1x1_layer4, upsampling1)

    # Deconv2
    upsampling2 = tf.layers.conv2d_transpose(skip1, num_classes, 4, strides=(2, 2), padding='same',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Predict3
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    skip2 = tf.add(conv_1x1_layer3, upsampling2)

    # Deconv3
    upsampling3 = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), padding='same',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))



```

## Result

Following three images are the result of different epoch number. First image is the result of epochs = 1, second is epochs = 20, third is epochs = 35. In this case, third image shows the most accurate result. 


Epoch1

![result_fig][image1]

Epoch20

![result_fig][image2]

Epoch 35

![result_fig][image3]






---
## Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

## Start
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

