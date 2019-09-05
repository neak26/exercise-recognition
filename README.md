# Exercise Recognition
## General information

We collected data from 25 persons; each person repeated each arm motion 5 times. We recorded video motion files separately, and extracted images from the video files. These images created our dataset for arm exercise recognition. We used transfer learning for for exercise recognition by retraining the final layer of Google's Inception-v3 model that is a deep convolutional architecture trained on the Imagenet dataset. The final layer of the Inception-v3 model was trained by using 80\% of the data of all the subjects for training, 10\% for validation and 10\% for testing. We have 6 classes: idle, arms-up, arms-side, arms-front, arms-front-bending, and arms-side-bending. 

recognition.py opens the Pepper robots camera and gets images from the robot's camera. It also loads the pre-trained models and labels. 

## How to use the program?
### Training with your own data:
Please follow the instructions in the [link](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3)
### Using the trained model with my classes:
- The trained files are given in the [link](https://www.dropbox.com/sh/8v5i616qppfdewr/AADK2wx-jG4Ivl2iyOJ5gGjQa?dl=0)
- Download the files and unzip in your computer. 
- Change the path of the files in recognition.py
- Run recognition.py by `python recognition.py`
- If you have not installed NAOqi SDK and you would like to use with Nao or Pepper, see installation instructions in the [link](http://doc.aldebaran.com/2-5/dev/python/install_guide.html)


