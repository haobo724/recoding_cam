import cv2
from liner_regression import get_regression
import numpy as np
from tools import delet_contours,block_analyse
def load_video(path1,path2):
    video_top = cv2.VideoCapture(path1)
    video_bot = cv2.VideoCapture(path2)
    return video_top,video_bot
def change_points(input_points):
    s = input_points.sum(axis=1)
    p1 = input_points[np.argmin(s)]
    p3 = input_points[np.argmax(s)]
    return (p1,p3)
    # diff = np.diff(input_points, axis=1)
    # p2 = input_points[np.argmin(diff)]
    # p4 = input_points[np.argmax(diff)]
    #
    # # 声明一个所有元素都为 0 的矩阵
    # rect = np.zeros((4, 2), dtype="float32")
    # rect[0] = p1
    # rect[1] = p2
    # rect[2] = p3
    # rect[3] = p4
    # print(rect)
    # return rect

def color( img):
    low_range = np.array([30, 40, 0])
    high_range = np.array([77, 255, 255])
    th = cv2.inRange(img, low_range, high_range)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    if len(contours) == 0:
        # print('No Green Point')
        return None, None, None
    delete_list = []
    e_list = []
    for i in range(len(contours)):
        l = len(contours[i])
        area = cv2.contourArea(contours[i])
        e = 4 * np.pi * area / (l * l)  # Rundheit
        if (abs(1.2 - e) > 0.25) or cv2.contourArea(contours[i]) < 15:
            delete_list.append(i)
        else:
            e_list.append(e)
    contours = delet_contours(contours, delete_list)


    if contours is None or len(contours) != 4:
        print(f'no markers')
        return [-1,-1,-1,-1]
    else:

        innerpst = np.array([])
        for c in contours:
            epsilon = 0.01 * cv2.arcLength(c, True)
            new_c = cv2.approxPolyDP(c, epsilon, True)
            x, y, w, h = cv2.boundingRect(new_c)
            pt = (int(x + w // 2), int(y + h // 2))  # 使用前三个矩m00, m01和m10计算重心
            np.append(innerpst,pt)
        return change_points(innerpst)
def get_green_point(video):
    point_list = []
    while True:
        ret,frame = video.read()
        if ret:
            point_list.append(color(frame))
        else:
            break



def get_height(video):
    height_list = []
    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.selectROI('frame_bot', frame, fromCenter=False)
    block_height = frame[y:y + h, x:x + w]
    height = block_analyse(block_height)
    height_list.append(height)
    while True:
        ret, frame = video.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            block_height = frame[y:y + h, x:x + w]
            height = block_analyse(block_height)
            height_list.append(height)

        else:
            break
    return height_list


def do_regression():
    pt1=get_regression()
    # pt2=get_regression()
    # pt3=get_regression()
    pt4=get_regression()