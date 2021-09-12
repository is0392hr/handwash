from utils import detector_utils as detector_utils
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf2
import datetime
from scipy.signal import savgol_filter
import argparse
import matplotlib.pyplot as plt
import csv
from pathlib import Path

from spring import spring

detection_graph, sess = detector_utils.load_inference_graph()
with open('./trajectory_x.csv') as f:
    reader = csv.reader(f)
    centers_x = [int(row[0]) for row in reader]

template1 = np.array(centers_x[270:300])
template1_vel = np.diff(template1)
template1_vel = savgol_filter(template1_vel, 11, 3)
template2 = np.array(centers_x[470:529])
template2_vel = np.diff(template2)
template2_vel = savgol_filter(template2_vel, 17, 3)
template3 = np.array(centers_x[470:529])
template3_vel = np.diff(template3)
template3_vel = savgol_filter(template3_vel, 11, 3)
template4 = np.array(centers_x[1126:1165])
template4_vel = np.diff(template4)
template4_vel = savgol_filter(template4_vel, 11, 3)

Y_ = [template1_vel, template2_vel, template3_vel, template4_vel]
E_ = [180, 1800, 2300, 3800] #[250, 2000, 2500, 3800]
pathes =[]   


def hand_detection(video_src, out_path, score_thresh, w, h, visualize):
    cap = cv2.VideoCapture(video_src)   #('/Users/kokiasami/Desktop/test1.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))  #('./output_detection_test1.mp4', fourcc, fps, (w, h))

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 1

    if visualize:
        cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    centers_x = []
    centers_y = []
    detected_boxes = []

    while cap.isOpened():
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
            break

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
        if ret == True:
            boxes, scores = detector_utils.detect_objects(image_np,
                                                              detection_graph, sess)

            # for box in boxes:
            #     print(box)

            # draw bounding boxes on frame
            center_x, center_y, detected_box = detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
                                             scores, boxes, im_width, im_height, image_np, visualize)
            centers_x.append(center_x)
            centers_y.append(center_y)
            detected_boxes.append(detected_box)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            detector_utils.draw_framecount_on_image(str(num_frames), image_np)
            detector_utils.draw_fps_on_image("FPS : " + str(int(fps)), image_np)

            # writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if (visualize > 0):
                cv2.imshow('Single-Threaded Detection',
                           cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q') or num_frames>1847:
                break
            else:
                pass
                # print("frames processed: ", num_frames, "elapsed time: ",
                #       elapsed_time, "fps: ", str(int(fps)))
        else:
            break

    cap.release()
    # writer.release()
    cv2.destroyAllWindows() 

    return  centers_x, centers_y, detected_boxes

def action_recognition(video_src, out_path, query, boxes, visualize):
    cap = cv2.VideoCapture(video_src)  # ('/Users/kokiasami/Desktop/test1.mp4')

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))  #('./output_AR_test1.mp4', fourcc, fps, (w, h))

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))

    if visualize:
        cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    ## Action Recognition
    query = np.array(query)
    query_vel = np.diff(query)
    query_vel = savgol_filter(query_vel, 11, 3)

    X = query_vel
    for Y, E in zip(Y_, E_):
        for path, cost in spring(X, Y, E):
            pathes.extend(path[:,0])

    data = np.zeros(len(query))
    for i in range(len(query)):
        if i in pathes:
            data[i] = 1
        else:
            data[i] = 0

    # drawing AR result
    while cap.isOpened() and num_frames < len(data):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
            break

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
        if ret == True:
            # draw bounding boxes on frame
            detected = detector_utils.draw_ARbbox_on_image(boxes[num_frames], im_width, im_height,
                                                                image_np, data[num_frames])

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            detector_utils.draw_framecount_on_image(str(num_frames), image_np)
            detector_utils.draw_fps_on_image("FPS : " + str(int(fps)), image_np)

            writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            if visualize:
                cv2.imshow('Single-Threaded Detection',
                           cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q') or num_frames>1847:
                break
            else:
                pass
                # print("frames processed: ", num_frames, "elapsed time: ",
                #       elapsed_time, "fps: ", str(int(fps)))
        else:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=640,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=480,
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
    args = parser.parse_args()
    print("Feature extraction...")
    name_proper1 = 'handwash_480p.mov'
    name_proper2 = 'test5.mp4'
    name_inproper1 = 'test_2.mp4'
    name_inproper1 = 'test4.mp4'
    name = name_proper2
    centers_x, centers_y, detected_boxes = hand_detection(
                                                            video_src = '/Users/kokiasami/Desktop/'+name, 
                                                            out_path = 'output_detection.mp4', 
                                                            score_thresh = args.score_thresh,
                                                            w = 640, 
                                                            h = 480, 
                                                            visualize = 0
                                                        )

    with open('./trajectory_x_'+name.split('.')[0]+'.csv', 'w') as f:
        writer = csv.writer(f)
        for cx in centers_x:
            writer.writerow([cx])

    print("Action recognition...")
    action_recognition(
                        video_src = '/Users/kokiasami/Desktop/'+name, 
                        out_path = 'output_AR_'+name, 
                        query = centers_x,
                        boxes = detected_boxes,
                        visualize = 1
                    )
