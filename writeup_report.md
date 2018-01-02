# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/cnn_arch_model.png "Final CNN model"
[image2]: ./examples/centre_lane_driving.png "Centre lane driving example"
[image3]: ./examples/recovery.gif "Recovery example"
[image4]: ./examples/flipped.png "Flipped Image"
[image5]: ./examples/histogram.png "Histogram of datapoints"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

Also included are two videos showing the trained model controlling the simulator:
* video.mp4 is the first course
* video_track2.mp4 is the second course

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I chose to use [NVIDIA's network architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png).

The model consists of a convolution neural network with an initial normalisation layer, 3 convolutional layers using 5x5 filter kernels, 2 convolutional layers using 3x3 filter kernels, then a flattening layer followed by 3 fully-connected layers.

I modified the architecture by removing the largest fully-connected layer and adding an additional layer at the end to reduce the output to a single value, representing the predicted steering angle.

The model includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer and cropped to remove unneeded image information such as the car bonnet and the sky/treeline to improve the speed and performance of the model. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The datasets include both available tracks, driving around the first track in both directions and adding some recovery data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I also added a small amount of dropout after each of the fully-connected layers.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and a combination of the three available mounted camera (central, left and right) to provide further data to the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep things simple. I used a modified version of a known architecture provided by NVIDIA and didn't venture too far with my modifications. I didn't add any overfitting techniques from the offset to see how the model would fair. I instead focused on collecting a representative dataset.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. From the off I found that the model was reaching about 5 to 7 epochs with a good reduction in my mean squared error for both training and validation. To combat overfitting, I never went beyond 7 epochs.

I gathered data by performing 1 full lap of both tracks, and combined this with the provided dataset. I added in a few recovery sets, driving from the edge of the track back to a central position. 

To further add to my dataset, I used the left and right mounted cameras with an offset to the steering angle and added vertically flipped images and inverted the steering measurements.

Finally, I noticed that there were a high percentage of images representing zero and close-to-zero steering angles, so I removed a random proportion to provide a more flattened histogram.

I played a little with adding derivate action to the existing PI controller, so that I could increase the driving speed of the car in autonomous mode. However, I couldn't find tuning parameters that kept the car on the road, so eventually I only increased the driving speed to around 14.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 1 lap on track one using center lane driving. Here is an example image of center lane driving (to the best of my ability!). It also shows the use of the three mounted cameras (left, centre and right) to provide more data, particularly capturing off-centre steering angles.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from any off-centre lane driving.

![alt text][image3]

Then I took 1 full lap from track two in order to get more data points.

To augment the data set, I flipped images and angles thinking that this would prevent having too many datapoints skewed towards using left steering angles. For example, here is an image that has then been flipped:

![alt text][image4]

---
After the collection process and data augmentation I had around 63,000 data points. This is how the data points were distributed across the steering angles.

![alt_text][image5]

Blue shows the original dataset, and orange shows the reduced data set used for training the model after compensating for the numerous low valued steering angles.


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. I used an adam optimizer so that manually training the learning rate wasn't necessary.
