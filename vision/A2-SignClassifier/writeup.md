# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/liu599/python-dpl-playground/blob/master/vision/A2-SignClassifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text](./examples/data.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it can save 3 times space and without an accuracy loss.

Here is an example of a traffic sign image before and after grayscaling.

![alt text](./examples/grayscale.png)

As a last step, I normalized the image data because the neural network will converge faster.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| input 400, output 120                         |
| RELU					|												|
| Fully connected		| input 120, output 84                          |
| RELU					|												|
| Fully connected		| input 84, output 43                           |
| Softmax				|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 60 epochs, batch size as 128, learning rate as 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.936
* test set accuracy of 0.926

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    - LeNet, it shows a good performance for classifier
* What were some problems with the initial architecture?
    - The initial one shows a low accuracy because I did not normalize my data. After I normalize the data, the model initializes with a 0.7 accuracy which is better.
* How was the architecture adjusted and why was it adjusted? 
    - Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
    - It shows a better accuracy when I increase epochs (1, 10, 40)
    - Learning rate performances best at 0.001, but not good for 0.0001.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    - Convolution layers are used for this kind of problem. They are more clever than the normal neural network. They have several different filters/kernels consisting of (randomly initialized) trainable parameters depending on the depth and filters at each layer of a network, which can convolve on a given input volume (the first input being the image itself) spatially to create some feature/activation maps at each layer.

If a well known architecture was chosen:
* What architecture was chosen?
    - LeNet
* Why did you believe it would be relevant to the traffic sign application?
    - This problem is an image recognition problem, LeNet is a CNN model which is good for this kind of problem.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    - my final model shows training set accuracy of 0.996, validation set accuracy of 0.936, test set accuracy of 0.926
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](./images/38.png) 
![alt text](./images/1.png)
![alt text](./images/21.png) 
![alt text](./images/12.png) 
![alt text](./images/13.png)

Going for each of the pictures, the last two signs, yield signs and priority road are easy to classify because the shape of these two signs are special and relatively unique. The round speed limit and keep right signs are very difficult to classify because there are a lot of signs in the same shape and they all look similar (The area of difference for these signs are relatively small).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| keep Right      		| Road work   									| 
| Speed limit (30km/h)  | Speed limit (50km/h)                          |
| double Curve			| Dangerous curve to the right					|
| priority road	      	| Priority road					 				|
| yield sign			| Yield              							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares not good to the accuracy on the test set of 93%. The model is hard to classify the speed limit sign like we discussed above. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work   									| 
| 1.0     				| Yield 										|
| 0.997, 0.002			| Children crossing, Dangerous curve to the right |
| 1.0	      			| Priority road					 				|
| 1.0				    | Yield      							        |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


