import math
import random

import numpy as np
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

hand_detector = HandDetector(detectionCon=0.8, maxHands=1)

class snakeGame:
    def __init__(self, pathFood):
        self.points = []
        self.distances = []
        self.current_length = 0
        self.allowed_length = 150
        self.previous_head = 0,0
        self.score = 0
        self.high_score = 0
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoints = 0,0
        self.random_food_loc()
        self.gameover = False


    def random_food_loc(self):
        self.foodPoints = random.randint(100, 1000), random.randint(100,600)

    def update(self, imgMain, currentHead):

        if self.gameover:
            cvzone.putTextRect(imgMain, "Game Over", [300,300], scale=4, thickness=5)
            cvzone.putTextRect(imgMain, f"Your Score: {self.score}", [300, 450], scale=4, thickness=5)
            cvzone.putTextRect(imgMain, f"Highest Score: {self.high_score}", [300, 550], scale=4, thickness=5)

        else:

            px, py = self.previous_head
            cx, cy = currentHead
            self.points.append([cx, cy])
            distance = math.hypot(cx-px, cy-py)
            self.distances.append(distance)
            self.current_length+=distance
            self.previous_head=cx,cy

            if self.current_length > self.allowed_length:
                for i, length in enumerate(self.distances):
                    self.current_length-=length
                    self.distances.pop(i)
                    self.points.pop(i)
                    if self.current_length < self.allowed_length:
                        break

            rx, ry = self.foodPoints
            if rx-self.wFood//2<cx<rx+self.wFood//2 and ry-self.hFood//2<cy<ry+self.hFood//2:
                self.random_food_loc()
                self.allowed_length+=50
                self.score+=1
                print(self.score)


            # Draw Snake
            if self.points:
                for i, point in enumerate(self.points):
                    if (i!=0):
                        cv2.line(imgMain, self.points[i-1], self.points[i], (0,0,255), 20)
                cv2.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx-self.wFood//2, ry-self.hFood//2))

            cvzone.putTextRect(imgMain, f"Score: {self.score}", [50, 80], scale=3, thickness=3, offset=10)


            # Check for collusion
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            min_distance = cv2.pointPolygonTest(pts, (cx,cy), True)

            if (-1<=min_distance<=1):
                self.gameover=True
                self.points = []
                self.distances = []
                self.current_length = 0
                self.allowed_length = 150
                self.previous_head = 0, 0
                self.random_food_loc()

                if (self.high_score <= self.score):
                    self.high_score = self.score

        return imgMain





game = snakeGame('Dont.png')


while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    hands, img = hand_detector.findHands(image, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameover=False
        game.score=0