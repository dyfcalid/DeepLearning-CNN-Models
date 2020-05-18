# DeepLearning-CNN-Models
  
## Introduction
This project is based on `TensorFlow 2` and has implemented representative convolutional neural networks in recent years, which are trained on the `CIFAR-10` dataset and suitable for image classification tasks. The basic architecture of the network refers to the **original papers** on `arXiv` as much as possible, and some of them have been modified for the CIFAR-10 dataset. The best accuracy is **96.60%**.
  
## Environment 
- Python 3.7  
- TensorFlow-gpu 2.1  
- Jupyter Notebook  
- GPU: **NVIDIA TESLA P100**  
  
## Papers
  
- **AlexNet** (2012) : [ImageNet Classification with Deep ConvolutionalNeural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
- **NetworkInNetwork** (2014) : [Network In Network](https://arxiv.org/pdf/1312.4400.pdf)  
- **VGG** (2014) : [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)  
- **InceptionV1** (2014) : [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842)  
- **InceptionV2, InceptionV3** (2015) :  
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)  
- **InceptionV4, InceptionResNet** (2016) : 
  - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261)  
- **ResNet** (2015,2016) :   
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)  
  - [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027v3.pdf)  
- **DilatedConvolution** (2016) : [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/pdf/1511.07122)  
- **SqueezeNet** (2016) : [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360v3)
- **Stochastic Depth** (2016) : [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382v1)
- **FractalNet** (2017) : [Ultra-Deep Neural Networks without Residuals](https://arxiv.org/pdf/1605.07648.pdf)  
- **Xception** (2017) : [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357)  
- **PyramidNet** (2017) : [Deep Pyramidal Residual Networks](https://arxiv.org/pdf/1610.02915) 
- **ResNeXt** (2017) : [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431)  
- **WideResNet** (2017) : [Wide Residual Networks](https://arxiv.org/pdf/1605.07146)  
- **DenseNet** (2017) : [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993)  
- **DualPathNet** (2017) : [Dual Path Networks](https://arxiv.org/pdf/1707.01629)  
- **ShuffleNet** :  
  - **V1** (2017) : [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083)
  - **V2** (2018) : [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164)
- **MobileNet** :  
  - **V1** (2017) : [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861)
  - **V2** (2018) : [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)
  - **V3** (2019) : [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244)
- **SENet** (2019) : [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
- **CBAM** (2018) : [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521)
- **SKNet** (2019) : [Selective Kernel Networks](https://arxiv.org/pdf/1903.06586)
- **EfficientNet** (2019) : [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946)
- **ResNeSt** (2020) : [ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955)  
- **Other** :  
  - **tricks** : [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187)
  - **NasNet** : [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012)
  - **AmoebaNet** : [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548)
  
## Result  
  
Dataset: CIFAR-10  
**No Pre-train**  
  
| Network               | Params   | Batch Size | Epochs | Time Per Epoch | Total Time  |   Accuracy  |   Remarks   |
|:----------------------|:--------:|:----------:|:------:|:--------------:|:-----------:|:-----------:|:-----------:|
| [AlexNet][1]          |  9.63M   |    128     |  100   |      36s       |   1h        |    78.44%   |             |
| [NIN][2]              |  0.97M   |    128     |  100   |      36s       |   1h        |    90.38%   |             |
| [VGG16][3]            |  33.69M  |    128     |  100   |      41s       |   1h 8min   |    92.34%   |             |
| [InceptionV1][4]      |  0.37M   |    128     |  100   |      42s       |   1h 10min  |    93.02%   |  simplified |
| [InceptionV2][5]      |  0.65M   |    128     |  100   |      51s       |   1h 25min  |    93.40%   |  simplified |  
| [InceptionV3][6]      |  1.17M   |    128     |  100   |      55s       |   1h 30min  |    94.20%   |  simplified |
| [InceptionV4][7]      |  2.57M   |    128     |  100   |      104s      |   2h 53min  |    94.55%   |  simplified |
| [ResNet18][8]         |  11.18M  |    128     |  150   |      39s       |   1h 38min  |    95.11%   |   pre-act   |  
| [ResNet50][9]         |  23.59M  |    128     |  100   |      88s       |   2h 27min  |    94.55%   |   pre-act   |
| [DilatedConv][10]     |  2.02M   |    128     |  100   |      92s       |   2h 33min  |    93.22%   |             |
| [SqueezeNet][11]      |  0.73M   |     32     |  100   |      35s       |   58min     |    88.41%   | light-weight|
| [StochasticDepth][12] |  23.59M  |    128     |  100   |      92s       |   2h 33min  |    95.07%   |   ResNet50  |
| [FractalNet][13]      |  33.76M  |    128     |  100   |      48s       |   1h 20min  |    94.32%   |             |  
| [Xception][14]        |  1.36M   |    128     |  100   |      54s       |   1h 30min  |    94.56%   |  simplified |  
| [PyramidNet110][15]   |  9.90M   |    128     |  100   |      185s      |   5h 8min   |  **95.65%** |             |  
| [ResNeXt50][16]       |  23.11M  |    128     |  100   |      210s      |   5h 50min  |    95.43%   |   32×4d     |  
| [WideResNet][17]      |  36.51M  |    128     |  150   |      138s      |   5h 45min  |  **95.94%** |   28-10     |  
| [DenseNet100][18]     |  3.31M   |    128     |  150   |      159s      |   6h 38min  |  **95.57%** |   100-24    |  
| [DenseNet121][19]     |  7.94M   |    128     |  100   |      110s      |   3h 3min   |    94.91%   |   121-32    | 
| [DualPathNet50][20]   |  21.05M  |    128     |  100   |      220s      |   6h 7min   |    95.44%   |             |  
| [DualPathNet92][21]   |  34.38M  |    128     |  100   |      370s      |   10h 17min |  **95.78%** |             |  
| [ShuffleNetV2][22]    |  1.28M   |    128     |  100   |      39s       |   1h 5min   |    92.41%   | light-weight|  
| [MobileNetV3][23]     |  4.21M   |    128     |  100   |      66s       |   1h 50min  |    94.85%   | light-weight|  
| [SE-ResNet50][24]     |  26.10M  |    128     |  100   |      110s      |   3h 3min   |    95.37%   |             |  
| [SE-ResNeXt50][25]    |  25.59M  |    128     |  120   |      270s      |   9h        |  **96.12%** |    32×4d    |  
| [SE-WideResNet][26]   |  36.86M  |    128     |  150   |      175s      |   7h 18min  |  **96.60%** |    28-10    |
| [SENet154][27]        |  567.9M  |    128     |  100   |      ----      |    -----    |    -----    |             |  
| [CBAM-ResNet50][28]   |  26.12M  |    128     |  100   |      154s      |   4h 17min  |    95.01%   |             |   
| [SKNet][29]           |  6.73M   |    256     |  100   |      205s      |    -----    |    -----    |             |   
| [EfficientNetB0][30]  |  3.45M   |    64      |  100   |      390s      |    -----    |    -----    |             | 
   
**SOTA : SE-WideResNet (Acc. : 96.60%)**  
  
Remarks:
 - simplified : replace the stem structure with one convolutional layer, channels are divided by 4
 - pre-act : ResNet V2 (full pre-activation)  
 - light-weight : smaller efficient CNN architecture which is suitable for mobile and embedded vision applications
   
## Implement Detail   
details of the SOTA network :
  - Architecture : SE-WideResNet (28-10)
  - Data augment
  - Learning rate
  
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
[11]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/SqueezeNet/cifar10_SqueezeNet.ipynb  
[12]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/StochasticDepth/cifar10_ResNet50_StochasticDepth.ipynb
[13]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/FractalNet/cifar10_FractalNet.ipynb
[14]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/Xception/cifar10_Xception.ipynb
[15]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/PyramidNet/cifar10_PyramidNet.ipynb  
[16]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/ResNeXt/cifar10_ResNeXt50.ipynb
[17]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/WideResNet/cifar10_WideResNet.ipynb
[18]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DenseNet/cifar10_DenseNet100.ipynb
[19]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DenseNet/cifar10_DenseNet121.ipynb
[20]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DualPathNet/cifar10_DualPathNet50.ipynb
[21]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/DualPathNet/cifar10_DualPathNet92.ipynb
[22]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/ShuffleNet/cifar10_ShuffleNetV2.ipynb
[23]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/MobileNet/cifar10_MobileNetV3.ipynb  
[24]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/SENet/cifar10_SE-ResNet50.ipynb
[25]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/SENet/cifar10_SE-ResNeXt50.ipynb
[26]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/SENet/cifar10_SE-WideResNet.ipynb
[27]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/SENet/cifar10_SENet154.ipynb
[28]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/CBAM/cifar10_CBAM-ResNet50.ipynb
[29]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/SKNet/cifar10_SKNet50.ipynb
[30]:https://nbviewer.jupyter.org/github/dyfcalid/DeepLearning-CNN-Models/blob/master/EfficientNet/cifar10_EfficientNetB0.ipynb  
  
  
