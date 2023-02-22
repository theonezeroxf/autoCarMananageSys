import cv2
import cv2 as cv
import numpy as np
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
#############################
path="E:/photo/car/jing2.jpg"
img_w=630
img_h=470
#############################

minArea=10
img=cv.imread(path)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
nPlateCascade =cv.CascadeClassifier("./Resource/haarcascade_russian_plate_number.xml")
numberPlates=nPlateCascade.detectMultiScale(imgGray,1.1,10)
print(len(numberPlates))
for (x,y,w,h) in  numberPlates:
    area=w*h
    if area > minArea:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # cv2.putText(img, "Number Plate", (x, y - 5),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        imgRoi = img[y:y + h, x:x + w]
        imgRoi=cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
        imgRoi=cv2.resize(imgRoi,(640,200))
        imageArray = ([img, imgGray,imgRoi])
        stackedImages = stackImages(0.6, imageArray)
        cv2.imshow("SucessFlow", stackedImages)
        cv2.waitKey(0)
        # cv2.imwrite("E:/photo/car/imgRoi.jpg",imgRoi)
    else:
        imageArray = ([img, imgGray])
        stackedImages = stackImages(0.6, imageArray)
        cv2.imshow("FailFlow", stackedImages)
        cv2.waitKey(0)