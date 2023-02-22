import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 1 直接以灰度图的方式读入
def getBinary_pic(path):
    """
    :param path:pic_path
    :return: binary_pic
    """
    img=cv.imread(path,0)
    histr=cv.calcHist([img],[0],None,[256],[0,256])
    # print(f"type={type(histr)},value={histr}")
    index=np.argmax(histr[100:255])
    print(index+100)
    ret, img2 = cv.threshold(img,(100+index)-10, 255, cv.THRESH_BINARY)
    # plt.figure(figsize=(10, 6), dpi=100)
    # plt.plot(histr)
    # plt.grid()
    # plt.show()
    return img2

if __name__=="__main__":
    img_binary=getBinary_pic("E:/pictures/cp/chars/other.bmp")
    cv.imshow("binaryImg",img_binary)
    cv.waitKey(0)