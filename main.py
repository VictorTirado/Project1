import numpy as np
import cv2


if __name__ == '__main__':

    image = input("Input image: ")
    template = input("Target image:")
    threshold = input("Detection threshold:")
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    template2 = cv2.imread(template,cv2.IMREAD_ANYCOLOR)
    cv2.imshow("Normal",img)
    cv2.imshow("Template",template2)

    imgFound = np.zeros((40,245,3),np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imgFound,"TARGET FOUND",(5,30),font,1,(0,255,0),2)

    imgNotFound = np.zeros((40,245,3),np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imgNotFound, "TARGET NOT FOUND", (5, 30), font, 1, (0, 255, 0), 2)

    cv2.imshow("Result",imgFound)
    cv2.imshow("Result", imgNotFound)

    cv2.waitKey(0)