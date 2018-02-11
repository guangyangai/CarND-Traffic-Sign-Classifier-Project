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


[//]: # (Image References)

[image1]: ./output_images/bar_count_signs.png "Visualization"
[image2]: ./output_images/sign_example2.png "Original"
[image3]: ./output_images/transformed_sign_example2.png "Grayscaling"
[image4]: ./output_images/german_test1.png "Traffic Sign 1"
[image5]: ./output_images/german_test2.png "Traffic Sign 2"
[image6]: ./output_images/german_test3.png "Traffic Sign 3"
[image7]: ./output_images/german_test4.png "Traffic Sign 4"
[image8]: ./output_images/german_test5.png "Traffic Sign 4"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/guangyangai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of signs for each type. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is really not helpful in recognizing the tyoe of the traffic sign. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because scaling is needed for classification for faster convergence. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3*64     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Dropout					|												|With probability 0.5
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5*64    | 1x1 stride, valid padding, outputs 10x10x64      									|
| RELU					|												|
| Dropout					|												|With probability 0.5
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected		| Input 1600, Output 120       									|
| Fully connected		| Input 120, Output 84       									|
| Fully connected		| Input 84, Output 10       									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer with a bath size of 1280. The number of epochs is 200 and the learning rate is set as 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.93 (Once reached in 0.94 in previous epoch) 
* test set accuracy of 0.912

If a well known architecture was chosen:
* What architecture was chosen? The LetNet-5 architecture from class is used. 
* Why did you believe it would be relevant to the traffic sign application? From the lecture, it is said LetNet-5 is a good starting point, althouhg I did not get an initial accuracy of 0.89. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation accuracy is promising. And the test set result is also good indicating that the model is overfit (possibly because I added in dropout layer)
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because it's very dark.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| No passing     			| No passing										|
| Road narrows on the right | Road narrows on the right										|
| Speed limit (80km/h)      		| Speed limit (80km/h)			 				|
| General caution		| General caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.912.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The result probabilies can be seen in the last cell (topk), surprising it differentiates pretty well except for third image.
For the first image, the model is relatively sure that this is a Keep right sign (probability of almost 1), and the image is a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .857         			| Speed limit (80km/h)									| 
| .135     				| Speed limit (50km/h)									|
| .0072					| Speed limit (100km/h)						|
| .0006	      			| Speed limit (20km/h)	 				|
| .00003			    | Speed limit (60km/h)   							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


