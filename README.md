# Real Time Hand Gesture Recognition and Tracking

### Pipeline:

1. Hand detection -> Transeferable model of ssd-mobilenet-v1-coco is trained for detecting hands. It takes images as input and outputs bounding box of detected hands. Dataset used : http://vision.soic.indiana.edu/projects/egohands/.

2. Static Hand Gesture Recognition: Grayscale cropped images of detected hands are passed through a CNN which classifies the gesture in static frames.
Required Dataset is generated using Hand detection model used in step one.

3. Hand tracking: Centroid tracking algorithm is used for tracking of detected hand.



### File Description:

1. train_detector_hand.ipynb -> training ssd-mobilenet (pre-trained) using transfer learning to detect hands (took approximately 50-60k iterations to reach total loss below 2.4).

2. hand_inference_graph2 -> saved tensorflow graph for ssd-mobilenet trained in above notebook.

3. addpose.py -> script used for creating dataset (live) for classification of custom gestures. Hands are detected from each frame webcam live stream (using above trained detector) are cropped to (64x64) and saved to a directory (./Gestures/New by default).

<img src="images/asl_f.png" width=10% title="asl_f"> <img src="images/fist.png" width=10% title="fist"> <img src="images/palm.png" width=10% title="palm"> <img src="images/seven.png" width=10% title="seven">   + garbage

3. recognition2.ipynb -> training the custom cnn for classification of static gestures. It took about 10 iterations to reach val_accuracy of 99.99%. 

Cnn architecture : 

  Layer (type)         |        Output Shape
  
  input_1 (InputLayer)         [(None, 32, 32, 1)]
  
  conv2d (Conv2D)              (None, 30, 30, 3)

  batch_normalization (BatchNo (None, 30, 30, 32)

  conv2d_1 (Conv2D)            (None, 28, 28, 64)

  batch_normalization_1 (Batch (None, 28, 28, 64)
  
  max_pooling2d (MaxPooling2D) (None, 14, 14, 64)

  dropout (Dropout)            (None, 14, 14, 64)

  flatten (Flatten)            (None, 12544)

  dense (Dense)                (None, 128)

  dropout_1 (Dropout)          (None, 128)

  dense_1 (Dense)              (None, 5)


    
  InputLayer [None, 32, 32, 3]
  
  Conv2D => BatchNormalization =>  Conv2D => BatchNormalization => MaxPooling2D => Dropout => Flatten => Dense => Dropout => Dense
  

4. #### detect.py and detect_multi.py : 

    Script to run detections and classification on detected hands. (single threaded and multi-threaded respectively). Multi-threading is used to increase the fps.     More : https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

5. utils contains some utility functions to run detections and classification.


------------------------------------------------------------------------------------

Reference: https://github.com/victordibia/handtracking
