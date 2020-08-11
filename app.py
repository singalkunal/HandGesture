from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
from utils.recognition_utils import preprocess_img
from utils.tracking_utils import *
import datetime
import argparse
import numpy as np
import pyautogui

score_thresh = 0.2

MEMORY_LIMIT = 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf
.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
    except RuntimeError as e:
        print(e)


# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue


# model = tf.keras.models.load_model('./saved_model')

mp = ['asl_f', 'fist', 'garbage', 'palm', 'seven']
mp2 = ['up', 'left', 'down', 'right']
IMG_SIZE =  32
def classify(img, model):
    img_np = preprocess_img(img, IMG_SIZE)

    preds = model.predict(np.expand_dims(img_np, axis=0))
    pred = np.argmax(preds)
    pred = pred.astype(int)

    # print(np.max(preds), " > ", mp[pred])
    return img_np, pred


def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    model = tf.keras.models.load_model('./saved_model2/')

    #################
    # all the following variables store data for sliding window
    sw_size = 30 # sliding window size
    trajecX = [] # trajectory x coord
    trajecY = [] # trajectory y coord
    preds = [] # classified classes 
    preds_freq = np.zeros(5) # frequency of each class
    #################

    screenWd, screenHt = pyautogui.size()
    frameWd, frameHt = cap_params['im_width'], cap_params['im_height']

    scaleX = 1.0 * screenWd / frameWd
    scaleY = 1.0 * screenHt / frameHt

    state = -1
    rad_in, rad_mid, rad_out= 25, 90, 200 # radii of annulus
    last_swipe, swipe_cnt = -1, 0 # will provide sanitary check so same swipes are not performed twice consecutively

    while True:
        # print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()

        if (frame is not None):
            # print(nm, "Not None")
            im_width = cap_params['im_width']
            im_height = cap_params['im_height']

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # get cropped image
            cropped_img = detector_utils.get_box_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)


            # draw bounding boxes
            centroidX, centroidY = detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)

            # negative centroid means no hand is detected
            if centroidX < 0:
                frame_processed += 1
                output_q.put(frame)
                # cropped_output_q.put(cropped_img)
                continue

            # Updation of centroid and add to trajectory if needed
            if centroidX > 0:
                if len(trajecX) == 0:
                    trajecX.append(centroidX)
                    trajecY.append(centroidY)
                else:
                    px, py = trajecX[-1], trajecY[-1]
                    d = dist(centroidX, centroidY, px, py)
                    if d <= rad_in or (d > rad_mid and d < rad_out):
                        centroidX, centroidY = px, py

                    trajecX.append(centroidX)
                    trajecY.append(centroidY)

                    
            # move sliding window
            if len(trajecX) > sw_size:
                trajecX = trajecX[-sw_size:]
                trajecY = trajecY[-sw_size:]


            # draw centroid
            if len(trajecX) > 0:
                x, y = trajecX[-1], trajecY[-1]

                # visualize_roi(cap_params['num_hands_detect'], (x, y), 
                        # cap_params['score_thresh'], scores, 
                        # rad_in, rad_mid, rad_out, frame)

                detector_utils.draw_centroid_on_image(
                    cap_params['num_hands_detect'], (x, y), cap_params["score_thresh"],
                    scores, frame)


            # Static gesture prediction
            if cropped_img is not None:
                cropped_img, idx = classify(cropped_img, model)
                detector_utils.draw_label_on_image(mp[idx], frame)
                preds.append(idx)
                preds_freq[idx] += 1

                if len(preds) > sw_size:
                    preds_freq[preds[0]] -= 1
                    preds = preds[-sw_size:]

                if preds[-1] == 1 or preds[-1] == 3:
                    state = control_vlc(last_swipe, preds[-1], state)


                # Control mouse pointer
                # if len(trajecX) > 1:
                    # x = int((trajecX[-1] - trajecX[-2]) )
                    # y = int((trajecY[-1] - trajecY[-2]) )
                    # control_mouse_pointer(x, y, screenWd, screenHt, preds[-1], preds[-2])

            # Dynamic gesture prediction
            if len(trajecX) > 0:
                draw_trajectory(trajecX, trajecY, frame)
                p = probab(preds_freq)
                d = dist(trajecX[-1], trajecY[-1], trajecX[0], trajecY[0])
                # print(d, p)

                if p > 0.5 and d > 110:
                    direc = get_direction(trajecX[-1], trajecY[-1], trajecX[0], trajecY[0])
                    if not last_swipe == direc:
                        # print("swipe {} count: ".format(mp2[last_swipe]), swipe_cnt)
                        swipe_cnt = 0
                    else:
                        swipe_cnt += 1

                    last_swipe = direc

                    if swipe_cnt % 6== 0 and swipe_cnt > 0:
                        print("swipe {} count: ".format(mp2[last_swipe]), swipe_cnt)
                        state = control_vlc(last_swipe, preds[-1], state)


                    detector_utils.draw_label_on_image(mp2[direc], frame, (500, 100))

                

            frame_processed += 1
            output_q.put(frame)
            # cropped_output_q.put(cropped_img)
        else:
            output_q.put(frame)
            # cropped_output_q.put(cropped_img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=1,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()


    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    cropped_output_q = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    #cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()
        # cropped_output_frame = cropped_output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)

        """"
        if (cropped_output_frame is not None):
            cropped_output_frame = cv2.cvtColor(cropped_output_frame.astype('float32'), cv2.COLOR_RGB2BGR)
            cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Cropped', 240, 240)
            cv2.imshow('Cropped', cropped_output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

        """

        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                cv2.imshow('Multi-Threaded Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))
        else:
            # print("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
