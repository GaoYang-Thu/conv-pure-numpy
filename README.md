# conv-pure-numpy

This repo builds a multi-layer convolutional neural network with numpy, and without any deep learning tools, including TF, Keras, etc.

## Comments
* Convolutional layers are established and tested using an example cat image. **No errors**.
* Currently: images could pass through multi-convolutional layers and result in feature maps (*they are just images, actually*).
* Next: 
    1. visualize all feature maps
    2. start from these feature maps, build full connected layers, and reach final output (*the output form is 2 numbers: \[a,b]*)

## *References*
1. [仅用numpy完成卷积神经网络](https://m.aliyun.com/yunqi/articles/585741)
2. [ahmedfgad's cnn-numpy project](https://github.com/ahmedfgad/NumPyCNN)
3. [Classifying Cats vs Dogs with a Convolutional Neural Network on Kaggle](https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/)
