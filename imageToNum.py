import cv2
import numpy as np
import pytesseract as pt
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
def preProcessing(img):

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    return imgBlur

img_path ="E:/pictures/cp/chars/other.bmp"
#******************* 读取图片为灰度格式并查看 ********************#
img = cv2.imread(img_path,0)
# img_Res=preProcessing(img)
#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.show()
ret,img2 = cv2.threshold(img,135,255,cv2.THRESH_BINARY_INV)
# cv2.resize(img2,(90,170))
stackedImages = stackImages(0.6, [[img,img2]])
cv2.imshow("WorkFlow", stackedImages)
cv2.waitKey(0)
# text = pt.image_to_string(img2,lang="chi_sim",config=r'  -psm  10')
text = pt.image_to_string(img2)

print(type(text),len(text))
print(text)
