from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream, normalize 
import datetime
import argparse
import numpy as np

frame_processed = 0
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

def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()

    while True:
        frame = input_q.get()
        if (frame is not None):
            im_width = cap_params['im_width']
            im_height = cap_params['im_height']

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)

            cropped_img = detector_utils.get_box_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)


            frame_processed += 1
            output_q.put(frame)
            cropped_output_q.put(cropped_img)
        else:
            output_q.put(frame)
            cropped_output_q.put(cropped_img)


if __name__ == '__main__':
    save = False

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
        default=1,
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
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    parser.add_argument(
            '-l',
            '--label',
            dest='label',
            type=str,
            default='img',
            help='label class of image'
            )
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

    path = './Pose/New/' + args.label + '_'
    count = 1
    print(path, args.label)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()
        cropped_output_frame = cropped_output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time

        if (cropped_output_frame is not None):
            cropped_output_frame = normalize(cv2.cvtColor(cropped_output_frame, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Cropped', 240, 240)
            cv2.imshow('Cropped', cropped_output_frame)
            if save:
                cv2.imwrite(path + str(count) + ".png", cropped_output_frame)
                count += 1
                if count == 1000:
                    print("1k examples generate....Press s to continue recording")
                    save = False
            if count >= 2200:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;


        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                cv2.imshow('Multi-Threaded Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    print("Pressed s")
                    save = not save
                    if save:
                        print("~~~ Recording gestures ~~~")
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
    print("{} images generated successfully..".format(count))
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
