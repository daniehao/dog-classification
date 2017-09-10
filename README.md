# Dog-Breed-Classification

## Project Introduction
This project aims to figure out a quick-way to match a human's photo to the closeast breed of dog based on CNN(conventional neural networks) and DNN(deep neural networks) models. I only hope to do an identify of the dog-breed identification at the beginning, but later on, I came up with this interesting idea to do a mapping with human's photo to a speicfic breed of dog in largest similarity. The following steps are included in my project.
Step 0: 
Import dog data
Step: 1:
Detect Humans
Step 2: 
Detect Dogs
Step 3: 
Create a CNN to Classify Dog Breeds(Transfer Learning)
Step 4:
Upload your picture

## Dataset
My dataset for this dog- classification project includes the images of 133 breed of dogs, which is collected by Udacity.com. The three dataset train, test and valid have already been classified. The train dataset is used for estimating predictive relationships and fit for the estimators; The valid dataset is used for improving overfitting problems existed in train dataset; The test dataset is used for testing the performance of the whole network.

## Motivation
* Earned some essential knowledges about CNN and DNN 
I learned about CNN(conventional neural networks) and DNN(deep neural networks) from online classes, which 

* kaggle cat vs dog competition. 
Few months ago, I saw a competition from Kaggle named "Create an algorithm to distinguish dogs from cats". Even though this competition was due by four years ago, this interesting topic is still the initial incentive for me to do this project and which enables me to take this into next level by doing dog-breed classifaction.

* Advanced and classifed dataset provided
Collecting data for different breed of dogs could be a tough work, since it requires 70 to 100 plus imagines for the same breed of dog from all over the world. However, the dataset I got has already been collected and classified, so I could directly use it for deep learning modeling.

* Well- defined architecture
OpenCV and Restnet50 have been pre-trained and well- defined.

## Third-partiy package I used
1.Keras

2.Tensorflow

3.Sklearn

4.numpy

## My approach
1.Human-face detector: OpenCV pre-trained haarcascade_frontalface classifier 

OpenCV pre-trained haarcascade_frontalface classifier helps to extract 2D and 3D features in python and works well on face detector. However, The type I error for my Human Face Detector is 4%, and the type II error for Human Face Detector is 11%. The dog defector performs better on my dataset with a zero type I error and a zero type II error.

2.Dog-face detector and Transfer learning: ResNet50 pre-trained model 

I mainly use Restnet50 for the dog detector, the architecture that was originaly developed by researchers from Microsoft Research. The paper Deep Residual Learning for Image Recognition talks about how does the Microsoft team design experiments on ImageNet to show the degradation problem and evaluate their method. Their achievement is to evaluate residual nets with a depth of up to 152 layers—8× deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. With a higher accuracy and well- defined architecture, it is earsier for me to do the transfer learning in the last part in order to achieve my final goal. The accuracy provided by their team is 3.57% from big picture, but I also did an accuracy measurement in my own project as I mentioned above.  

## Next steps
1.More data (why more data can help?)
### [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)

2.Better architecture to decrease the type I and type II error of OpenCV.

3.Web-based app
I hope to do a web-based app and could enable any people to use this program.
