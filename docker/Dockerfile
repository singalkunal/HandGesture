#FROM singalkunal/tensorflow-opencv:tf-gpu-cv4
FROM local/tensorflow-opencv:tf-gpu-cv4

#install tk gui toolkit
RUN apt-get update \
    && apt-get install python3-tk

WORKDIR HandGesture

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY hand_inference_graph2 hand_inference_graph2/
COPY object_detection object_detection/
COPY saved_model2 saved_model2/
COPY utils utils/
COPY app.py .

RUN rm requirements.txt

#ENTRYPOINT ["python3", "app.py"]
#CMD [`]

