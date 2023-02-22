import cv2
import numpy as np
import pytesseract as pt
###################################
widthImg=640
heightImg =200
# path_mask="E:/photo/car/mask.jpg"
path="E:/photo/car/m1.jpg"
####################################
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

def getCp(img):
    nPlateCascade = cv2.CascadeClassifier("./Resource/haarcascade_russian_plate_number.xml")
    numberPlates = nPlateCascade.detectMultiScale(img, 1.1, 10)
    len_numPlate= len(numberPlates)
    if len_numPlate==0:
        img = cv2.resize(img, (640, 200))
        return 0,img
    else:
        (x, y, w, h)=numberPlates[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        imgRoi = img[y:y + h, x:x + w]
        imgRoi = cv2.resize(imgRoi, (640, 200))
        return 1,imgRoi

def getMask(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([98, 149, 147])
    upper = np.array([120, 255, 253])
    mask = cv2.inRange(imgHSV, lower, upper)
    return mask

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    # imgCanny = cv2.Canny(imgBlur,200,200)
    # kernel = np.ones((5,5))
    # imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    # imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgBlur

def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped

def getContours(img):
    # ret, img2 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    imgCanny = cv2.Canny(img, 50, 50)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count = 1
    maxArea = 0
    biggest = np.array([])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f"count={count},area={area}")
        if area > 1000:
            # cv2.drawContours(img_Copy, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
            count += 1
    x, y, w, h = cv2.boundingRect(biggest)
    img=getWarp(img,biggest)
    cv2.rectangle(img_Copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img=img[y:y + h, x:x + w]
    img=cv2.resize(img,(widthImg,heightImg))
    return img

def getString(img):
    ret, img2 = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY_INV)
    cv2.resize(img2, (90, 170))
    text = pt.image_to_string(img2, lang="chi_sim", config=r'  -psm  10')

    return text

img=cv2.imread(path)

ret,img_cp=getCp(img)
img_Copy=img_cp

img_Res=preProcessing(img_cp)

img_Contour=getContours(img_Res)


str=getString(img_Contour)
imageArray = ([img,img_cp],[img_Copy,img_Contour])
stackedImages = stackImages(0.6, imageArray)
print(str)
cv2.imshow("WorkFlow", stackedImages)
cv2.waitKey(0)
