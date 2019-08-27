# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/dataset_bar_chart.png "barchart"
[image2]: ./examples/dataset_samples.png "samples"
[image3]: ./examples/sample.png "sample"
[image4]: ./examples/sample_grayscaled.png "sample_gray"
[image5]: ./examples/sample_equalized.png "sample_equalized"
[image6]: ./examples/aug_sample.png "aug_sample"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

#### 1. Submission Files
I use the workspace for this project, the submission files listed as below:
* __Ipython notebook with code__: _Traffic_Sign_Classifier.ipynb_ in the project directory
* __HTML output of the code__: _Traffic_Sign_Classifier.html_ in the project directory
* __A writeup report__: _writeup.md_ in the project directory

### Dataset Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32x32x3`
* The number of unique classes/labels in the data set is `43` (apply np.unique() to `y_train`)

#### 2. Include an exploratory visualization of the dataset.

* The first figure is the 3 bar-charts for training/validations/test datasets respectively, showing the samples count of each label in the dataset 

![alt text][image1]

* While the second figure including 43 sample images, one sample for one label, selected from the training dataset 
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* The first step is to pre-process the original image, steps as below:
1. Step 1: convert it to gray-scale

   * The original image
![alt text][image3]

   * The grayscaled image
![alt text][image4]

2. Step 2: Apply histogram equlization to the grayscaled image, enhance the contrast, alleviate the impact of situations like too dark or too shining.![alt text][image5]

* The second step is to augment the original training dataset, because the sample counts of some lables are too small for training, and it's relatively easy to augment image dataset. I use the techniques provided in the Multiscale-CNN paper by Sermanet, details as below:

1. Step 1: Translate the orignal image, horizontally or vertically, randomly in range [-2, +2] pixels
2. Step 2: Rescale the whole image, by a ratio in the range from 0.9 to 1.1
3. Step 3: Rotate the image, by a degree in the range from -15 to +15 
4. Step 4: Decide how many images to augment for different label, I wish the final count of each label should be approximately in range from 2000 to 3000, logics as below:

    * The count of augmentation depends on the original dataset grouped by classes
    * If original class count > 1000, augment 1 image for every image
    * If original class count > 500 and < 1000, augment 3 images ..
    * If original class count < 500, augment 6 images ..

Here is an example of an augmented image:

![alt text][image6]

I printed out a brief summay of the augmented dataset as below:
Count of augmentation x: 73197

Count of augmentation y: 73197

Label: 0, Count: 1260

Label: 1, Count: 3960

Label: 2, Count: 4020

Label: 3, Count: 2520

Label: 4, Count: 3540

Label: 5, Count: 3300

Label: 6, Count: 2520

Label: 7, Count: 2580

Label: 8, Count: 2520

Label: 9, Count: 2640

Label: 10, Count: 3600

Label: 11, Count: 2340

Label: 12, Count: 3780

Label: 13, Count: 3840

Label: 14, Count: 2760

Label: 15, Count: 2160

Label: 16, Count: 2520

Label: 17, Count: 3960

Label: 18, Count: 2160

Label: 19, Count: 1260

Label: 20, Count: 2100

Label: 21, Count: 1890

Label: 22, Count: 2310

Label: 23, Count: 3150

Label: 24, Count: 1680

Label: 25, Count: 2700

Label: 26, Count: 2160

Label: 27, Count: 1470

Label: 28, Count: 3360

Label: 29, Count: 1680

Label: 30, Count: 2730

Label: 31, Count: 2760

Label: 32, Count: 1470

Label: 33, Count: 2396

Label: 34, Count: 2520

Label: 35, Count: 2160

Label: 36, Count: 2310

Label: 37, Count: 1260

Label: 38, Count: 3720

Label: 39, Count: 1890

Label: 40, Count: 2100

Label: 41, Count: 1470

Label: 42, Count: 1470



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


