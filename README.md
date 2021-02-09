# Real Time Hand Gesture Recognition and Tracking

### Human Computer Interaction (HCI)

Real time robust hand gestures recognition and tracking on a live stream . This forms the basis for HCI (Human Computer Interaction). Users can interact with the computer (e.g. control vlc media player or control mouse pointer  here) via webcam using dynamic hand gestures accomplished using computer vision (CNN’s)

### This branch provides containerized solution using docker

### Usage:

<b>1. run prebuilt image</b>

    $ wget https://raw.github.com/singalkunal/HandGesture/container/scripts/run.sh
    $ chmod +x run.sh
    $ ./run.sh      # takes time when run for the first time

    $./run.sh -x    # to see available optional command line args


<b>2. build locally</b>

    $ wget https://raw.github.com/singalkunal/HandGesture/container/scripts/build.sh
    $ chmod +x build.sh
    $ ./build.sh        # local/handgesture:latest image will be generated

    $ wget https://raw.github.com/singalkunal/HandGesture/container/scripts/run.sh
    $ chmod +x run.sh
    $ ./run.sh -l       # -l flag tells tu run local image

