# dog-breed-classification

## Project Introduction
This project aims to figure out a quick-way to match a human's photo to the closeast breed of dog based on CNN(conventional neural networks) and DNN(deep neural networks) models. I only hope to do an identify of the dog-breed identification at the beginning, but later on, I came up with this interesting idea to do a mapping with human's photo to a speicfic breed of dog in largest similarity.

## Dataset
My dataset for this dog- classification project includes the images of 133 breed of dogs, which is collected by Udacity.com. The three dataset train, test and valid have already been classified. The train dataset is used for estimating predictive relationships and fit for the estimators; The valid dataset is used for improving overfitting problems existed in train dataset; The test dataset is used for testing the performance of the whole network.

## Motivation
1. Earned some essential knowledges about CNN and DNN 
I learned about CNN(conventional neural networks) and DNN(deep neural networks) from online classes, which 

II. kaggle cat vs dog competition. 
Few months ago, I saw a competition from Kaggle named "Create an algorithm to distinguish dogs from cats". Even though this competition was due by four years ago, this interesting topic is still the initial incentive for me to do this project and which enables me to take this into next level by doing dog-breed classifaction.

II. Advanced and classifed dataset provided
Collecting data for different breed of dogs could be a tough work, since it requires 70 to 100 plus imagines for the same breed of dog from all over the world. However, the dataset I got has already been collected and classified, so I could directly use it for deep learning modeling.

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

