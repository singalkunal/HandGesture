# Real Time Hand Gesture Recognition and Tracking

### Pipeline:

1. Hand detection -> Transeferable model of ssd-mobilenet-v1-coco is trained for detecting hands. It takes images as input and outputs bounding box of detected hands. Dataset used : http://vision.soic.indiana.edu/projects/egohands/.

2. Static Hand Gesture Recognition: Grayscale cropped images of detected hands are passed through a CNN which classifies the gesture in static frames.
Required Dataset is generated using Hand detection model used in step one.

3. Hand tracking: Centroid tracking algorithm is used for tracking of detected hand.


Files:

1. train_detector_hand.ipynb -> training ssd-mobilenet (pre-trained) using transfer learning to detect hands (took approximately 50-60k iterations to reach total loss below 2.4).

2. hand_inference_graph2 -> saved tensorflow graph for ssd-mobilenet trained in above notebook.

3. addpose.py -> script used for creating dataset (live) for classification of custom gestures. Hands are detected from each frame webcam live stream (using above trained detector) are cropped to (64x64) and saved to a directory (./Gestures/New by default).

<img src="images/asl_f.png" width=10%> <img src="images/fist.png" width=10%> <img src="images/palm.png" width=10%> <img src="images/seven.png" width=10%> 

asl_f   fist    palm    seven

3. recognition2.ipynb -> training the custom cnn for classification of static gestures.

