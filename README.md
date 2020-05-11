# DeepLearning-CNN-Models
  
## Introduction
This project is based on **TensorFlow 2** and has implemented representative convolutional neural networks in recent years, which are trained on the **CIFAR-10** dataset and suitable for image classification tasks. The basic architecture of the network refers to the original **papers on arXiv** as much as possible, and some of them have been modified for the CIFAR-10 dataset.
  
## Environment 
- **Python 3.7**  
- **TensorFlow-gpu 2.1**  
- **Jupyter Notebook**  
- **GPU: NVIDIA TESLA P100**  
  
## Papers
  
- **AlexNet** : [ImageNet Classification with Deep ConvolutionalNeural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **NetworkInNetwork** : [Network In Network](https://arxiv.org/pdf/1312.4400.pdf)
- **VGG** : [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)
- **InceptionV1** : [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842)
- **InceptionV2, InceptionV3** :  
  [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)
- **InceptionV4** : [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261)
- **ResNet** : [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
  
## Result  
  
| Network               | Params   | Batch Size | Epochs | Time Per Epoch | Accuracy(%) |   Remarks   |
|:----------------------|:--------:|:----------:|:------:|:--------------:|:-----------:|:-----------:|
| [AlexNet][1]          |  9.63M   |    128     |  100   |      36s       |    78.44    |             |
| [NIN][2]              |  0.97M   |    128     |  100   |      36s       |    90.38    |             |
| [Vgg16][3]            |  33.69M  |    128     |  100   |      41s       |    92.34    |             |
| [InceptionV1][4]      |  0.37M   |    128     |  100   |      42s       |    93.02    |  simplified |
| [InceptionV2][5]      |  0.65M   |    128     |  100   |      51s       |    93.40    |  simplified |  
| [InceptionV3][6]      |  1.17M   |    128     |  100   |      55s       |    94.20    |  simplified |
| [InceptionV4][7]      |  2.57M   |    128     |  100   |      104s      |    94.55    |  simplified |
| [ResNet18][8]         |  11.18M  |    128     |  150   |      39s       |    95.11    |   pre-act   |  
| [ResNet50][9]         |  23.59M  |    128     |  100   |      88s       |    94.55    |   pre-act   |

  
## License  
[MIT License](LICENSE)

  
  
  
  
[1]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/AlexNet/cifar10_AlexNet.ipynb
[2]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/NetworkInNetwork/cifar10_NIN.ipynb
[3]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/VGG/cifar10_VGG16.ipynb
[4]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/GoogLeNet/cifar10_InceptionV1.ipynb
[5]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/GoogLeNet/cifar10_InceptionV2.ipynb
[6]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/GoogLeNet/cifar10_InceptionV3.ipynb
[7]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/GoogLeNet/cifar10_InceptionV4.ipynb
[8]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/ResNet/cifar10_ResNet18.ipynb
[9]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/ResNet/cifar10_ResNet50.ipynb
  
