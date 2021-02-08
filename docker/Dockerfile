FROM singalkunal/tensorflow-opencv:tf-gpu-cv4
RUN apt-get update \
    && apt-get install vim -y

WORKDIR HandGesture

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY hand_inference_graph2 hand_inference_graph2/
COPY object_detection object_detection/
COPY saved_model2 saved_model2/
COPY utils utils/
COPY app.py .

RUN rm requirements.txt
