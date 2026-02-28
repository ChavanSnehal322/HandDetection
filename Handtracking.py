
import cv2
import mediapipe as mp
import  time

cap  = cv2.VideoCapture(0)

pTime = 0

# Creating object for handdetection module

mpHands = mp.solutions.hands            # default line to initiate the module

#  Calling default hand function
hands = mpHands.Hands( )

#  drawing the landmarks of the hand points using default function
mpDraw = mp.solutions.drawing_utils

while True:

    success, img = cap.read()

    #   Converting to img to RGB scale, as hands use only RGB images by default
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #  Processing the img
    res = hands.process(imgRGB)

    #  Open the object extract the multiple hands ancd check if the hands are detected
    # print(res.multi_hand_landmarks)

    if res.multi_hand_landmarks:

        for handsLmks in res.multi_hand_landmarks:

    #          function to dected points of hand
            for id, lmk in enumerate( handsLmks.landmark):
                # print(id, lmk)

                h, w, c = img.shape
                cx, cy = int(lmk.x * w), int(lmk.y*h)
                print( id, cx, cy)

                # if id == 0: # ( highlights palm )
                #     cv2.circle(img,( cx, cy), 15, (255, 0, 255), cv2.FILLED)

                #  highlighting the 21 points of hand
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    #          Displays the points and connections of the hand fingures
            mpDraw.draw_landmarks(img, handsLmks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / ( cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow("Image", img)

    cv2.waitKey(1)