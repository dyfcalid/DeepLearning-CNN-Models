# DeepLearning-CNN-Models
  
## Introduction
This project is based on `TensorFlow 2` and has implemented representative convolutional neural networks in recent years, which are trained on the `CIFAR-10` dataset and suitable for image classification tasks. The basic architecture of the network refers to the **original papers** on `arXiv` as much as possible, and some of them have been modified for the CIFAR-10 dataset.
  
## Environment 
- Python 3.7  
- TensorFlow-gpu 2.1  
- Jupyter Notebook  
- GPU: **NVIDIA TESLA P100**  
  
## Papers
  
- **AlexNet** : [ImageNet Classification with Deep ConvolutionalNeural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
- **NetworkInNetwork** : [Network In Network](https://arxiv.org/pdf/1312.4400.pdf)  
- **VGG** : [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)  
- **InceptionV1** : [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842)  
- **InceptionV2, InceptionV3** :  
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)  
- **InceptionV4, InceptionResNet** : 
  - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261)  
- **ResNet** :   
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)  
  - [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027v3.pdf)  
- **DilatedConvolution** : [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/pdf/1511.07122)  
- **FractalNet** : [Ultra-Deep Neural Networks without Residuals](https://arxiv.org/pdf/1605.07648.pdf)  
- **Xception** : [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357)  
- **PyramidNet** : [Deep Pyramidal Residual Networks](https://arxiv.org/pdf/1610.02915) 
- **ResNeXt** : [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431)  
- **WideResNet** : [Wide Residual Networks](https://arxiv.org/pdf/1605.07146)
- **DenseNet** : [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993)    
- **DualPathNet** : [Dual Path Networks](https://arxiv.org/pdf/1707.01629)   
- **SqueezeNet**  : [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360v3)
- **SENet** : [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
- **CBAM** : [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521)
- **SKNet** : [Selective Kernel Networks](https://arxiv.org/pdf/1903.06586)
- **EfficientNet** : [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946)
  
  
## Result  
  
Dataset: CIFAR-10
  
| Network               | Params   | Batch Size | Epochs | Time Per Epoch | Accuracy(%) |   Remarks   |
|:----------------------|:--------:|:----------:|:------:|:--------------:|:-----------:|:-----------:|
| [AlexNet][1]          |  9.63M   |    128     |  100   |      36s       |    78.44    |             |
| [NIN][2]              |  0.97M   |    128     |  100   |      36s       |    90.38    |             |
| [VGG16][3]            |  33.69M  |    128     |  100   |      41s       |    92.34    |             |
| [InceptionV1][4]      |  0.37M   |    128     |  100   |      42s       |    93.02    |  simplified |
| [InceptionV2][5]      |  0.65M   |    128     |  100   |      51s       |    93.40    |  simplified |  
| [InceptionV3][6]      |  1.17M   |    128     |  100   |      55s       |    94.20    |  simplified |
| [InceptionV4][7]      |  2.57M   |    128     |  100   |      104s      |    94.55    |  simplified |
| [ResNet18][8]         |  11.18M  |    128     |  150   |      39s       |    95.11    |   pre-act   |  
| [ResNet50][9]         |  23.59M  |    128     |  100   |      88s       |    94.55    |   pre-act   |
| [DilatedConv][10]     |  2.02M   |    128     |  100   |      92s       |    93.22    |             |
| [FractalNet][11]      |  33.76M  |    128     |  100   |      48s       |    94.32    |             |  
| [Xception][12]        |  1.36M   |    128     |  100   |      54s       |    94.56    |  simplified |  
| [PyramidNet][13]      |  2.38M   |    128     |  100   |      100s      |    94.92    |             |  
| [ResNeXt50][14]       |  23.11M  |    128     |  100   |      210s      |    95.43    |             |  
| [WideResNet][15]      |  36.51M  |    128     |  150   |      138s      |    95.94    |   28-10     |  
| [DenseNet100][16]     |  3.31M   |    128     |  150   |      159s      |    95.57    |   100-24    |  
| [DenseNet121][17]     |  7.94M   |    128     |  100   |      110s      |    94.91    |   121-32    | 
| [DualPathNet50][18]   |  21.05M  |    128     |  100   |      220s      |    95.44    |             |  
| [DualPathNet92][19]   |  34.38M  |    128     |  100   |      370s      |    95.78    |             |  
| [SqueezeNet][20]      |  0.73M   |     32     |  100   |      35s       |    88.41    |  efficient  |  
  
remarks:
 - simplified : replace the stem structure with one convolutional layer, channels are divided by 4
 - pre-act : ResNet V2 (full pre-activation)  
 - efficient : smaller efficient CNN architecture which is suitable for mobile and embedded vision applications
   
   
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
[10]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DilatedConvolution/cifar10_DilatedConvolution.ipynb
[11]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/FractalNet/cifar10_FractalNet.ipynb
[12]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/Xception/cifar10_Xception.ipynb
[13]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/PyramidNet/cifar10_PyramidNet.ipynb  
[14]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/ResNeXt/cifar10_ResNeXt50.ipynb
[15]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/WideResNet/cifar10_WideResNet.ipynb
[16]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DenseNet/cifar10_DenseNet100.ipynb
[17]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DenseNet/cifar10_DenseNet121.ipynb
[18]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DualPathNet/cifar10_DualPathNet50.ipynb
[19]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DualPathNet/cifar10_DualPathNet92.ipynb
[20]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/SqueezeNet/cifar10_SqueezeNet.ipynb  
  
  
