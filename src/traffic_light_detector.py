import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # RED
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    # GREEN
    lower_green = np.array([40,70,70])
    upper_green = np.array([90,255,255])

    # YELLOW
    lower_yellow = np.array([15,100,100])
    upper_yellow = np.array([35,255,255])

    red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    green = cv2.inRange(hsv, lower_green, upper_green)
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    if cv2.countNonZero(red) > 500:
        cv2.putText(frame,"RED LIGHT",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    elif cv2.countNonZero(yellow) > 500:
        cv2.putText(frame,"YELLOW LIGHT",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)

    elif cv2.countNonZero(green) > 500:
        cv2.putText(frame,"GREEN LIGHT",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.imshow("Traffic Light Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

