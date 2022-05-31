import cv2
from liner_regression import get_regression
import numpy as np
from tools import delet_contours,block_analyse


orb = cv2.ORB_create()

def load_video(path1,path2):
    video_top = cv2.VideoCapture(path1)
    print('totol Frames :',video_top.get(cv2.CAP_PROP_FRAME_COUNT))
    video_bot = cv2.VideoCapture(path2)
    assert video_top.get(cv2.CAP_PROP_FRAME_COUNT) == video_bot.get(cv2.CAP_PROP_FRAME_COUNT)
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
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    low_range = np.array([100, 43, 46])
    high_range = np.array([124, 255, 255])
    th = cv2.inRange(img, low_range, high_range)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)

    blank = np.zeros_like(img)
    cv2.drawContours(blank, contours, -1, (255, 255, 255), -1)
    cv2.imshow('hi', blank)
    cv2.waitKey()

    if len(contours) == 0:
        print('No Green Point')
        return [-1,-1,-1,-1]
    delete_list = []
    e_list = []


    for i in range(len(contours)):
        l = len(contours[i])
        area = cv2.contourArea(contours[i])
        e = 4 * np.pi * area / (l * l)  # Rundheit
        if (abs(1.2 - e) > 0.25) :
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
    ret, frame = video.read()
    x, y, w, h = cv2.selectROI('roi',frame)
    cut = frame[y:y + h, x:x + w]
    bf = cv2.BFMatcher()
    kp,des =orb.detectAndCompute(cut,None)
    while True:
        ret,frame = video.read()
        if ret:
            cv2.imshow('hi', frame)
            cv2.waitKey()
            kp2, des2 = orb.detectAndCompute(frame, None)

            pt = color(frame)
            if not pt[0]== -1:
                point_list.append(pt)
        else:
            break
    print(len(point_list))


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

if __name__ == '__main__':
    path_top = 'patient0_top.mp4'
    path_bot = 'patient0_bot.mp4'
    v_t,v_b = load_video(path_top,path_bot)
    get_green_point(v_t)