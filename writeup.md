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
[image7]: ./examples/images_web_download.png "web images"
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

I didn't normalized the images because it would surprisingly decrease the validation accuracy if I use the Lenet-5 model to train, and the numerical stability looks good during the training process, so I commentted it out.

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
I tried to read the Multiscale-CNN paper by Sermanet, but cannot figure out what the exactly model they use, the model I designed in __Multiscale_CNN()__ method works not as good as expected, the validation accuracy always less than 90%. Thus, I still chose the Lenet-5 model and made a little change to finish the classfication task, and it works well with the augmented training dataset.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 5x5 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 5x5 stride, valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected 1		| 400x120 weights, outputs (120,)        									|
| RELU					|												|
| Dropout	      	| keep_prob = 0.5 				|
| Fully connected	2	| 120x84 weights, outputs (84,)        									|
| RELU					|												|
| Dropout	      	| keep_prob = 0.5 				|
| Fully connected	3	| 84x43 weights, outputs (43,)        									|
| RELU					|												|
| Dropout	      	| keep_prob = 0.5 				|
| Softmax				| based on the final output logits (43,)     									|
|	Cross Entropy			|	 to measure the loss											|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
The Lenet-5 model above will output logits with a shape of (43,), then we need to calculate the probabilities and cross-entropy together using __tf.nn.softmax_cross_entropy_with_logits()__ function, the loss function was simply defined by the mean of total cross entropy. Thus, in order to train the model, I used an __tf.train.AdamOptimizer__ to reduce the loss function iteratively, hyperparameters as below:
* EPOCHS = 50
* BATCH_SIZE = 128
* learning_rate = 0.0004
* keep_prob = 0.5 in training process, and keep_prob = 1.0 in validation process

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 95.9% 
* test set accuracy of 94.3%


As the project instructions suggested, first I went to read the Multiscale-CNN paper by Sermanet, the technique they introduced is brilliant and the accuracy approximately 99% is also excellent. But through out the paper, I cannot find the exactly model architecture they used, then I spent a lot of time to design the network based on my understanding, and tune parameters, however, unfortunately it works not as good as the Lenet-5 model. So, finally I still use the Lenet-5 to finish task


At the beginning, with the raw Lenet-5 model (no change made) and original training dataset (no preprocess or augmentation), it can generate a validation accuracy of 89%. Then I decided to preprocess the images, using grayscale/histogram equalization/normalization step by step, but something weird happen that the validation accuracy decreased to about 80%, the root cause is the normalization step, so I remove this step in preprocess. 


Train the model again, the validation accuracy increase a little to about 91%, it's not enough still. Then I began to augment the training dataset, because the distribution of samples with different labels is very unbalanced, so I attempt to augment them, mainly focused on those labels with a small sample count, to make every labels with an evenly distributed sample counts about 3000. With the augmented dataset, the accuracy improved a lot, reach to 96%


Finally, I evaluated the model on test dataset, the test accuracy is about 90%, it looks like the generalization performance is not so good, so I add a dropout step into the Lenet-5 fully connected layers to alleviate overfitting problem. Now the test accurancy is about 94%.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web, and then resize them to (32x32x3):

![alt text][image7]

The 3rd image might be difficult to classify because it's similar to children crossing images

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution      		| General Caution  									| 
| No Passing     			| No Passing										|
| End of no passing					| Children crossing											|
| Speed limit (60km/h)	      		| Speed limit (60km/h)					 				|
| Stop			| Stop      							|
| Ahead only			| Ahead only      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. Despite it's less than the test dataset accuracy, it's possibly because that the sample volume is too small, and label 41 is relatively more difficult to predict.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For all the 5 correctly predicted images, the most certain probability is 1.0, 100% sure about the prediction. The detail top 5 probabilities of the correctly predicted images were printed out in the Jupyter notebook, below is the top 5 probs of the 3rd image which is wrongly predicted:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.562        			| Children crossing   									| 
| 0.2558     				| Bumpy road 										|
| 0.1462					| Bicycles crossing											|
| 0.0359	      			| End of no passing					 				|
| 0.0000				    | No passing      							|
 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


