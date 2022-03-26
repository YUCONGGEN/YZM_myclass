# -*- coding=GBK -*-
import cv2 as cv
import os
pic_dir="./image"

#图像二值化 0白色 1黑色
#全局阈值
def threshold_image(image,labelx):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # print(gray)
    cv.imwrite("./image_ok/{}.jpg".format(labelx),gray)
    # cv.imshow("原来", gray)
    # cv.waitKey(1000)

    #ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)#大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    # print("阈值：%s" % ret)
    # cv.imshow("OTSU", binary)
    #
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)#TRIANGLE法,，全局自适应阈值, 参数0可改为任意数字但不起作用，适用于单个波峰
    # print("阈值：%s" % ret)
    # cv.imshow("TRIANGLE", binary)

    #ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)# 自定义阈值为150,大于150的是白色 小于的是黑色
    # print("阈值：%s" % ret)
    # cv.imshow("自定义", binary)
    # cv.waitKey(5000)

    v,binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)# 自定义阈值为150,大于150的是黑色 小于的是白色
    # print("阈值：%s" % ret)
    # cv.imshow("自定义反色", binary)
    #
    # ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TRUNC)# 截断 大于150的是改为150  小于150的保留
    # print("阈值：%s" % ret)
    # cv.imshow("截断1", binary)
    #
    # ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TOZERO)# 截断 小于150的是改为150  大于150的保留
    # print("阈值：%s" % ret)
    return binary


if __name__ == '__main__':
    file_name = [lable for lable in os.listdir(pic_dir)]
    for lablex in file_name:
        image_path = os.path.join(pic_dir, lablex)
        src = cv.imread(image_path)
        cv.imshow("截断2", src)
        cv.waitKey(10000)
        binary=threshold_image(src,lablex)
        cv.imwrite("./image_ok1/{}.jpg".format(lablex), binary)
        # cv.imwrite(int(binary),"./image_final")
        # cv.imshow("截断2", binary)
        # cv.waitKey(1000)