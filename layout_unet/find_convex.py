import cv2
from PIL import Image
import numpy as np
from post_process import thresholding


# 水平方向投影
def hProject(binary):
    h, w = binary.shape

    # 水平投影
    hprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建h长度都为0的数组
    h_h = [0]*h
    for j in range(h):
        for i in range(w):
            if binary[j,i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 255
    return hprojection


def find_convex(image_mask):
    img = np.zeros((image_mask.shape[0], image_mask.shape[1], 3), dtype=np.uint8)
    img += 255
    im_binary = thresholding(image_mask)
    hp = img.copy()
    hp = cv2.cvtColor(hp, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(im_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 3:  # A polygon cannot have less than 3 points
            continue
        
        # 剔除面积小于100的区域
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        # 凸包检测
        convex = cv2.convexHull(contour)
        print(len(contour))
        cv2.drawContours(img, [convex], 0, (0,255,0), 1)
        cv2.drawContours(img, [contour], 0, (255,0,0), 1)


        cv2.fillPoly(hp, [contour], (0, 0, 0))
        Image.fromarray(hp).show()
        minrect = cv2.boundingRect(contour)
        print(minrect)
        
        if area / (minrect[2]*minrect[3]) > 0.8:
            cv2.rectangle(img, (minrect[0], minrect[1]), (minrect[0]+minrect[2], minrect[1]+minrect[3]), (0,0,255), 1)
            continue
        img_contour = im_binary[minrect[1]:minrect[1]+minrect[3], minrect[0]:minrect[0]+minrect[2]]
        hp = hProject(img_contour)
        Image.fromarray(hp).show()
    Image.fromarray(img).show()
    


if __name__ == "__main__":
    
    img = Image.open('test.png')
    im = img.convert('L')

    im = np.array(im)

    # 选取区域类型对应的image
    im = np.where(im==1, 1, -1).astype(np.uint8)
    find_convex(im)