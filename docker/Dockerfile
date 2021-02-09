FROM singalkunal/tensorflow-opencv:latest-tf-gpu-cv4

RUN apt-get update
#    \ && apt-get install vim -y

WORKDIR HandGesture
COPY requirements.txt .

RUN pip3 install -r requirements.txt 

# install advance gui toolkit (Tkinter)
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata python3-tk

COPY hand_inference_graph2 hand_inference_graph2/
COPY object_detection object_detection/
COPY saved_model2 saved_model2/
COPY utils utils/
COPY app.py .

RUN rm requirements.txt 

ENTRYPOINT ["python3", "app.py"]
# CMD provides args to ENTRYPOINT wich can be overridden during runtime
CMD ["-nhands", "1", "-wd", "600", "-ht", "600"]
