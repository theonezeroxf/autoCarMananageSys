import cv2
import numpy as np
import os

# 分割结果输出路径
output_dir = "E:/pictures/cp/chars"
# 车牌路径
file_path="E:/pictures/cp/1.jpg"

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
# 循环操作每一张车牌

src = cv2.imread(file_path)
img = src.copy()

# 预处理去除螺丝点
cv2.circle(img, (145, 20), 10, (255, 0, 0), thickness=-1)
cv2.circle(img, (430, 20), 10, (255, 0, 0), thickness=-1)
cv2.circle(img, (145, 170), 10, (255, 0, 0), thickness=-1)
cv2.circle(img, (430, 170), 10, (255, 0, 0), thickness=-1)
cv2.circle(img, (180, 90), 10, (255, 0, 0), thickness=-1)

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 333, 1)

# 闭运算
kernel = np.ones((5, 5), int)
morphologyEx = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

# 找边界
contours, hierarchy = cv2.findContours(morphologyEx, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 画边界
img_1 = img.copy()
cv2.drawContours(img_1, contours, -1, (0, 0, 0), -1)

# imgStack = stackImages(0.7, ([src, img, gray], [adaptive_thresh, morphologyEx, img_1]))
# cv2.imshow("imgStack", imgStack)
# cv2.waitKey(0)

# 转灰度为了方便切割
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

# 每一列的白色数量
white = []
# 每一列的黑色数量
black = []
# 区域高度取决于图片高
height = gray_1.shape[0]
# 区域宽度取决于图片宽
width = gray_1.shape[1]
    # 最大白色数量
white_max = 0
# 最大黑色数量
black_max = 0
# 计算每一列的黑白色像素总和
for i in range(width):
    s = 0  # 这一列白色总数
    t = 0  # 这一列黑色总数
    for j in range(height):
        if gray_1[j][i] == 255:
            s += 1
        if gray_1[j][i] == 0:
            t += 1
    white_max = max(white_max, s)
    black_max = max(black_max, t)
    white.append(s)
    black.append(t)


# 找到右边界
def find_end(start):
    end = start + 1
    for m in range(start + 1, width - 1):
            # 基本全黑的列视为边界
        if black[m] >= black_max * 0.95:  # 0.95这个参数请多调整，对应下面的0.05
            end = m
            break
    return end


# 临时变量
n = 1

# 起始位置
start = 1

# 结束位置
end = 2

# 分割结果数量
num=0


# 从左边网右边遍历
while n < width - 2:
    n += 1
    # 找到白色即为确定起始地址
    # 不可以直接 white[n] > white_max
    if white[n] > 0.05 * white_max:
        start = n
        # 找到结束坐标
        end = find_end(start)
        # 下一个的起始地址
        n = end
        print(num)
        # 确保找到的是符合要求的,过小不是车牌号
        if end - start > 10:
            print(num)
            # 分割
            char = gray_1[1:height, start - 5:end + 5]
            char = cv2.resize(char, (32, 40), interpolation=cv2.INTER_CUBIC)
            # 保存分割结果到文件
            cv2.imwrite(output_dir+'/' + str(num) + '.jpg',char)
            num+=1


