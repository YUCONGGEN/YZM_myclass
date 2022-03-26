# -*- coding=GBK -*-
import cv2 as cv
import os
pic_dir="./image"

#ͼ���ֵ�� 0��ɫ 1��ɫ
#ȫ����ֵ
def threshold_image(image,labelx):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # print(gray)
    cv.imwrite("./image_ok/{}.jpg".format(labelx),gray)
    # cv.imshow("ԭ��", gray)
    # cv.waitKey(1000)

    #ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)#���ɷ�,ȫ������Ӧ��ֵ ����0�ɸ�Ϊ�������ֵ���������
    # print("��ֵ��%s" % ret)
    # cv.imshow("OTSU", binary)
    #
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)#TRIANGLE��,��ȫ������Ӧ��ֵ, ����0�ɸ�Ϊ�������ֵ��������ã������ڵ�������
    # print("��ֵ��%s" % ret)
    # cv.imshow("TRIANGLE", binary)

    #ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)# �Զ�����ֵΪ150,����150���ǰ�ɫ С�ڵ��Ǻ�ɫ
    # print("��ֵ��%s" % ret)
    # cv.imshow("�Զ���", binary)
    # cv.waitKey(5000)

    v,binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)# �Զ�����ֵΪ150,����150���Ǻ�ɫ С�ڵ��ǰ�ɫ
    # print("��ֵ��%s" % ret)
    # cv.imshow("�Զ��巴ɫ", binary)
    #
    # ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TRUNC)# �ض� ����150���Ǹ�Ϊ150  С��150�ı���
    # print("��ֵ��%s" % ret)
    # cv.imshow("�ض�1", binary)
    #
    # ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_TOZERO)# �ض� С��150���Ǹ�Ϊ150  ����150�ı���
    # print("��ֵ��%s" % ret)
    return binary


if __name__ == '__main__':
    file_name = [lable for lable in os.listdir(pic_dir)]
    for lablex in file_name:
        image_path = os.path.join(pic_dir, lablex)
        src = cv.imread(image_path)
        cv.imshow("�ض�2", src)
        cv.waitKey(10000)
        binary=threshold_image(src,lablex)
        cv.imwrite("./image_ok1/{}.jpg".format(lablex), binary)
        # cv.imwrite(int(binary),"./image_final")
        # cv.imshow("�ض�2", binary)
        # cv.waitKey(1000)