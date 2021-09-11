# -*- coding: utf-8 -*-
import time
import RPi.GPIO as GPIO

from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

detection_graph, sess = detector_utils.load_inference_graph()
# 超音波センサー(HC-SR04)制御クラス
class Sensor():
    def __init__(self):
        self.__TRIG = 19 # 物理番号19
        self.__ECHO = 21 # 物理番号21
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD) # 物理ピン番号で指定
        GPIO.setup(self.__TRIG,GPIO.OUT)
        GPIO.setup(self.__ECHO,GPIO.IN)

    def getDistance(self):
        GPIO.output(self.__TRIG, GPIO.LOW)
        # TRIG = HIGH
        GPIO.output(self.__TRIG, True)
        # 0.01ms後に TRIG = LOW
        time.sleep(0.00001)        
        GPIO.output(self.__TRIG, False)

        signaloff=0
        signalon=0
        # 発射時間
        while GPIO.input(self.__ECHO) == 0:
            signaloff = time.time()
        # 到着時間
        while GPIO.input(self.__ECHO) == 1:
            signalon = time.time()
        # 距離計算
        return (signalon - signaloff) * 17000

    def __del__(self):
        GPIO.cleanup()

def hand_detection(ini_time):
    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Hand wash status', cv2.WINDOW_NORMAL)

    count = 0

    while (time.time() - ini_time) <= 30:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        count += detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyWindow('Single-Threaded Detection')
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))

        cv2.destroyWindow('Single-Threaded Detection')


def main():

    sensor = Sensor()
    while True:
        distance = sensor.getDistance()
        print("{:.0f}cm".format(distance))

        if distance < 10:
            ini_time = time.time()
            hand_detection(ini_time)
        time.sleep(0.1)
    del sensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.3,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=0,
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
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
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
    

    main()