# Exercise Recognition
## General information

We collected data from 25 persons; each person repeated each arm motion 5 times. We recorded video motion files separately, and extracted images from the video files. These images created our dataset for arm exercise recognition. We used transfer learning for for exercise recognition by retraining the final layer of Google's Inception-v3 model that is a deep convolutional architecture trained on the Imagenet dataset. The final layer of the Inception-v3 model was trained by using 80\% of the data of all the subjects for training, 10\% for validation and 10\% for testing. We have 6 classes: idle, arms-up, arms-side, arms-front, arms-front-bending, and arms-side-bending. 

## How to use the program?
### Training with your own data:
Please follow the instructions in the [link](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3)
### Using the trained model with my classes:
The trained files are given in 
