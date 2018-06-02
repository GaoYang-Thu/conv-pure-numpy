# conv-pure-numpy

This repo builds a multi-layer convolutional neural network with numpy, and without any deep learning tools, including TF, Keras, etc.

## About my CNN
* Network structure:
    * input -> 1<sup>th</sup> conv layer -> 2<sup>nd</sup> conv layer -> fully connect layer -> output label
    * *(inspired by [Zhang Zhifei's work](https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf))*
* Details of each layer:
    * input image size: `50 by 50`
    * filter pile of 1<sup>th</sup> convolutional layer: `6` filters, size of each = `11 by 11`
    * threshold pile after 1<sup>th</sup> convolutional layer: `6` threshold arrays, size of each = `(50-11+1 =) 40 by 40`, each array has all equal values
    * output of 1<sup>th</sup> convolutional layer: `6` images, size of each = `20 by 20`   
    * filter group of 2<sup>nd</sup> convolutional layer: `6` piles, in each pile: `12` filters, size of each = `5 by 5`
    * threshold pile after 2<sup>nd</sup> convolutional layer: `12` threshold arrays, size of each = `(20-5+1 =) 16 by 16`, each array has all equal values
    * output of 2<sup>nd</sup> convolutional layer: `12` images, size of each = `8 by 8`
    * column-wise vectorization followed by concatenation  
    * input array of fully conncect layer: length = `12 * 8 * 8 = 768`
    * weights array of fully connect layer: `2 * 768`
    * final output: `1 * 2` array

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
6. [Zhang Zhifei's work _(PDF)_ on Derivation of Backpropagation in (*a rather simple*) Convolutional Neural Network](https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf)
7. [Stanford CS231n class on CNN](http://cs231n.github.io/convolutional-networks/)
