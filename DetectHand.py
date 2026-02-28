#  Using the hand detection module in this file directly


import cv2
import  time
import HandTrackingModule as HTM


cap = cv2.VideoCapture(0)
detectorObj = HTM.detectHand()

pTime = 0
cTime = 0

while True:

    success, img = cap.read()
    img = detectorObj.detect_Hand_lmks(img)
    lmList = detectorObj.detectPosition(img)

    if len(lmList) != 0:
        #  print the point only at index 4
        print(lmList[2])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #  store the 21 points of hand in a list
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)
