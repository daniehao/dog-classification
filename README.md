# dog-breed-classification

## Project Introduction
This project aims to figure out a quick-way to match a human's photo to the closeast breed of dog based on CNN(conventional neural networks) and DNN(deep neural networks) models. I only hope to do an identify of the dog-breed identification at the beginning, but later on, I came up with this interesting idea to do a mapping with human's photo to a speicfic breed of dog in largest similarity.

## Dataset
My dataset for this dog- classification project includes the images of 133 breed of dogs, which is collected by Udacity.com. The three dataset train, test and valid have already been classified. The train dataset is used for estimating predictive relationships and fit for the estimators; The valid dataset is used for improving overfitting problems existed in train dataset; The test dataset is used for testing the performance of the whole network.

## Motivation
1. I learned about CNN(conventional neural networks) and DNN(deep neural networks) from online classes, which 

2. kaggle cat vs dog competition. this project take this into next level by doing dog-breed classifaction

3. dataset itself is great. i can do soemthing with it.

## Third-partiy package I used
1.Keras

2.Tensorflow

3.Sklearn

4.numpy

## My approach
1.Human-face detector: OpenCV pre-trained haarcascade_frontalface classifier 
(u can also talk about type I, type II erros here)

2.Dog-face detector: ResNet50 pre-trained model using imageNet data from Keras

3.Transfer learning using resnet 50


## Next steps
1.More data (why more data can help?)
### [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)

2.Better architecture

3.Web-based app

