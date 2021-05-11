# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys
import random
import time

# import RPi.GPIO as GPIO

# from utils import detector_utils as detector_utils
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import datetime
import argparse

import qrcode

#detection_graph, sess = detector_utils.load_inference_graph()

'''
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
'''
class Virus(object):
        def __init__(self, width, height, scale):
                self.x = int(width/2 + random.randint(-50,30) - 35)
                self.y = int(height/2 - random.randint(0, height/2))
                self.width = width
                self.height = height
                self.vel_x = random.randint(-1,1)
                self.vel_y = random.randint(-1,1)
                self.scale = scale
                self.visible = True


        def draw(self,gameDisplay):
            self.move()
            if self.visible:               
                virus = pygame.image.load("img/virus_corona.png").convert_alpha()
                size = virus.get_rect()
                gameDisplay.blit(pygame.transform.scale(virus, (int(size[2]*self.scale), int(size[3]*self.scale))), (self.x, self.y))


        def move(self):
            self.x += self.vel_x
            self.y += self.vel_x

def detect_hand(score_thresh, fps, video_source, width, height, display, num_workers, queue_size, cap):
    cap = cap
    ini_time = datetime.datetime.now()
    start_time = datetime.datetime.now()
    num_frames = 0
    global count
    count = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    # while  (datetime.datetime.now() - ini_time).total_seconds() < 30:
    '''
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
        detected = detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        yield count 
        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
        '''
    return 1

def animation(score_thresh, fps, video_source, width, height, display, num_workers, queue_size, cap):
    count = 0
    pygame.init() # 初期化
    (w, h) = (281, 300)
    (x, y) = (w/2, h/2)
    GameDisplay = pygame.display.set_mode((w, h), 0, 32)
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Pygame Test") # ウィンドウの上の方に出てくるアレの指定
    
    num_virus = 15
    HP = 280
    scale = random.uniform(0.1,0.2)
    FINISH = False
    ini_time = datetime.datetime.now()

    while (datetime.datetime.now() - ini_time).total_seconds() < 20:
        count += 10*detect_hand(score_thresh=score_thresh, fps=fps, video_source=video_source, width=width, height=height, display=display, num_workers=num_workers, queue_size=queue_size, cap=cap)
        print(count)
        screen.fill((0, 0, 0, 0)) # 背景色の指定。RGBのはず

        bg = pygame.image.load("img/pose_kyosyu_hand.png").convert_alpha() # 背景画像の指定
        rect_bg = bg.get_rect() # 画像のサイズ取得？？だと思われる
        rect_bg[0] -= 30
        screen.blit(bg, rect_bg) # 背景画像の描画

        pygame.draw.rect(GameDisplay, (255,255,255), (w-50, 10, 30, HP), width=5)
        pygame.draw.rect(GameDisplay, (0,0,0), (w-50, 10, 30, HP))

        if HP-count > HP*0.5:
            score = 0
            pygame.draw.rect(GameDisplay, (0,255,0), (w-50, 10+count, 30, HP-count))
        elif HP-count > HP*0.2:
            score = 30
            num_virus = 10
            scale = random.uniform(0.1,0.15)
            pygame.draw.rect(GameDisplay, (255,110,0), (w-50, 10+count, 30, HP-count))
        elif HP-count > 0:
            score = 100
            scale = random.uniform(0.05,0.08)
            num_virus = 5
            pygame.draw.rect(GameDisplay, (255,0,0), (w-50, 10+count, 30, HP-count))
        else:
            FINISH = True
            qr = qrcode.QRCode()
            qr.add_data(score)
            qr.make()
            img = qr.make_image()
            img.save('QR_img/test.png')
            count = 0

        if FINISH:   
            if count == 0:
                screen.fill((0, 0, 0, 0)) # 背景色の指定。RGBのはず
                bg = pygame.image.load("img/pose_kyosyu_hand.png").convert_alpha() # 背景画像の指定
                rect_bg = bg.get_rect() # 画像のサイズ取得？？だと思われる
                rect_bg[0] -= 30
                screen.blit(bg, rect_bg) # 背景画像の描画
                pygame.draw.rect(GameDisplay, (255,255,255), (w-50, 10, 30, HP), width=5)
                pygame.draw.rect(GameDisplay, (0,0,0), (w-50, 10, 30, HP))
                pygame.time.wait(300) # 更新間隔
                pygame.display.update() # 画面更新
    
            elif count < 15:
                screen.fill((0, 0, 0, 0))
                bg = pygame.image.load("img/jidou_tearai_syoudoku.png").convert_alpha() # 背景画像の指定
                rect_bg = bg.get_rect()
                screen.blit(pygame.transform.scale(bg, (w,h)), rect_bg) # 背景画像の描画
                pygame.time.wait(1000) # 更新間隔
                pygame.display.update() # 画面更新

            else:
                screen.fill((0, 0, 0, 0))
                bg = pygame.image.load("QR_img/test.png").convert_alpha() # 背景画像の指定
                rect_bg = bg.get_rect()
                screen.blit(pygame.transform.scale(bg, (w,h)), rect_bg) # 背景画像の描画
                pygame.time.wait(1000) # 更新間隔
                pygame.display.update() # 画面更新
            
        
        else:
            [Virus(w, h, scale).draw(GameDisplay) for _ in range(num_virus)] 

            pygame.time.wait(300) # 更新間隔。多分ミリ秒
            pygame.display.update() # 画面更新


        for event in pygame.event.get(): # 終了処理
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

def main(score_thresh, fps, video_source, width, height, display, num_workers, queue_size, cap):
    # sensor = Sensor()
    while True:
        sensor = random.randint(0,30)
        print("sensor_val: ", sensor)
        if sensor < 15:
            animation(score_thresh=score_thresh, fps=fps, video_source=video_source, width=width, height=height, display=display, num_workers=num_workers, queue_size=queue_size, cap=cap)

    

if __name__ == "__main__":
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

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    main(score_thresh=0.2, fps=1, video_source=-1, width=320, height=180, display=1, num_workers=4, queue_size=5, cap=cap)