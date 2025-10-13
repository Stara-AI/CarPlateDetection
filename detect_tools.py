# encoding:utf-8
import cv2
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import csv
import os

# fontC = ImageFont.truetype("Font/platech.ttf", 20, 0)

# 绘图展示
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def drawRectBox(image, rect, addText, fontC, color=(0,0,255)):
    """
    绘制矩形框与结果
    :param image: 原始图像
    :param rect: 矩形框坐标, int类型
    :param addText: 类别名称
    :param fontC: 字体
    :return:
    """
    # 绘制位置方框
    cv2.rectangle(image, (rect[0], rect[1]),
                 (rect[2], rect[3]),
                 color, 2)

    # 绘制字体背景框
    # cv2.rectangle(image, (rect[0] - 1, rect[1] - 50), (rect[2], rect[1]), color, -1, cv2.LINE_AA)
    # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细.无法正常显示中文
    # cv2.putText(image, addText, (int(rect[0])+2, int(rect[1])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 可以显示中文
    # 字体自适应大小
    font_size = int((rect[3]-rect[1])/1.5)
    fontC = ImageFont.truetype("Font/platech.ttf", font_size, 0)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((rect[0]+2, rect[1]-font_size), addText, (0, 0, 255), font=fontC)
    imagex = np.array(img)
    return imagex


def img_cvread(path):
    # 读取含中文名的图片文件
    # img = cv2.imread(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def draw_boxes(img, boxes):
    for each in boxes:
        x1 = each[0]
        y1 = each[1]
        x2 = each[2]
        y2 = each[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img



def cvimg_to_qpiximg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    qpix_img = QPixmap(qimg)
    return qpix_img




# 封装函数:图片上显示中文
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



