# conv-pure-numpy

This repo builds a multi-layer convolutional neural network with numpy, and without any deep learning tools, including TF, Keras, etc.

## Comments
* Convolutional layers are established and tested using an example cat image. **No errors**.
* Forward pass through cnn is so slow...
* Currently: 
    1. images could pass through multi-convolutional layers and result in feature maps (*they are just images, actually*).
    2. start from these feature maps, build full connected layers, and reach final output (*the output array has 2 numbers: \[a,b]*).
* Next: 
    1. visualize all feature maps.
    2. build back-probagation of the convolutional layers and fully connection layers.

## *References*
1. [仅用numpy完成卷积神经网络](https://m.aliyun.com/yunqi/articles/585741)
2. [ahmedfgad's cnn-numpy project](https://github.com/ahmedfgad/NumPyCNN)
3. [Classifying Cats vs Dogs with a Convolutional Neural Network on Kaggle](https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/)
4. [Back Propagation in Convolutional Neural Networks — Intuition and Code](https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199)
5. [mayankgrwl97's cnn-backward code](https://gist.github.com/mayankgrwl97/7c85ed1cf353be7764e2fa8b010da4d3)
6. [This awesome PDF on Derivation of Backpropagation in (*a rather simple*) Convolutional Neural Network](https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf)
