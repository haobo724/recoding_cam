import os
import queue

import cv2
import numpy as np

template_dir = 'OCR_template'
img_template = []
if os.path.exists(template_dir):
    for i in range(10):
        img_file_path = os.path.join(template_dir, str(i) + '.jpg')
        t = cv2.imread(img_file_path, 0)
        t = cv2.resize(t, (40, 80))
        print(np.max(t))
        img_template.append(t)
        print(f'[INFO] TEMPLATE {i} lOADED')
else:
    raise FileNotFoundError('NO TEMPLATE')



class Buffer():
    def __init__(self,size):
        self.q  = queue.Queue(size)
        self.list = []

    def __getitem__(self,num):
        if num > len(self.list):
            raise IndexError('list index out of range')
        return self.list[num]


    def append(self,num):
        if not self.q.full():
            self.q.put(num)
            self.list.append(num)
        else:
            self.q.get()
            self.q.put(num)
            del self.list[0]
            self.list.append(num)
    def mean(self):
        if self.q.empty():
            return 0
        return np.mean(self.list)
    def most(self):
        vals, counts = np.unique(self.list, return_counts=True)
        index = np.argmax(counts)
        return vals[index]
def order_points_new( pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0, 1] != leftMost[1, 1]:
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    else:
        leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
    (tl, bl) = leftMost
    if rightMost[0, 1] != rightMost[1, 1]:
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    else:
        rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
    (tr, br) = rightMost
    # print(tl, tr, bl, br)
    x = tr[0] - tl[0]
    y = br[1] - tr[1]
    return np.array([tl, tr, bl, br], dtype="float32"), int(x), int(y)

def crop_block( thresh, x, y):
    # crop each block of display by cut the pixel range.


    '''

    -------------Y
    |
    |
    |
    X
    '''
    width = int(1 / 9 * x)
    block1 = thresh[int(1.5 / 50 * y): int(4.5 / 18 * y), int(1.9 / 18 * x): int(1.9 / 18 * x) + width]

    block2 = thresh[int(1.5 / 50 * y): int(4.5 / 9 * y), int(7 / 18 * x): int(7 / 18 * x) + 2*width]
    bx, by = block2.shape
    block2 = block2[0: int(bx / 2), 0: int(by)]

    block3 = thresh[int(1.5 / 50 * y): int(4.5 / 9 * y), int(13 / 18 * x): int(13 / 18 * x) + 2*width]
    bx3, by3 = block3.shape
    block3 = block3[0: int(bx3 / 2), 0: by3]

    block4 = thresh[int(11 / 18 * y): int(15 / 18 * y), int(5 / 18 * x): int(13 / 18 * x)]
    return block1, block2, block3, block4
def delet_contours(self, contours, delete_list):
    # delta作用是offset，因为del是直接pop出去，修改长度了
    delta = 0
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours

def block_analyse(imfrag):
    # new method of reading digits in the imfrag
    _, imfrag_h = imfrag.shape
    ret2, imfrag = cv2.threshold(imfrag, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi_size =  (40, 80)
    # detect single digit and detect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素

    # convert gray value for contour detection
    cnts, _ = cv2.findContours(imfrag.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digitCnts = []
    xloc = np.array([])
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print(h)
        # if height is more than 50, then digit is detected
        if h > 30 / 85 * imfrag_h:
            digitCnts.append(c)
            xloc = np.append(xloc, x)

    # if no connected component is detected, return ''
    if digitCnts == []:
        return -1
    # sort using x direction
    idx = np.argsort(xloc)
    tmp = digitCnts.copy()
    digitCnts = []
    for i in idx:
        digitCnts.append(tmp[i])

    digit = ''
    if len(digitCnts) > 3 or len(digitCnts)<1:
        print('detect error,Suggested click restart btn')
        return -1
    # print(len(digitCnts))
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = imfrag[y:y + h, x:x + w]

        if roi is not None:
            roi = cv2.resize(roi,roi_size)
            # cv2.imshow('roi',roi)
            # cv2.waitKey()
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

            acc = np.zeros(10)
            for i in range(10):
                acc[i] = get_match_score(roi, img_template[i])
            if np.max(acc) < 0.8:
                print(acc)
                digit += '-1'
            else:
                digit += str(np.argmax(acc))
        else:
            digit = 0
    try :
        result = int(digit)
    except ValueError:
        result = -1
    if result >200:
        return -1
    else:
        print(result)
    return result

def get_match_score(img, template):
    # print(np.max(template))
    tp = (img == 255) == (template == 255)
    tn = (img == 0) == (template == 0)
    fp = (img == 255) == (template == 0)
    fn = (img == 0) == (template == 255)

    # score =( np.sum(tp) + np.sum(tn)) / (np.sum(tp) + np.sum(tn) + np.sum(fp) + np.sum(fn))
    score =( np.sum(tp) + np.sum(tn)- np.sum(fp) - np.sum(fn))
    return score
if __name__ == '__main__':
    buffer = Buffer(5)
    buffer.append(10)
    buffer.append(10)
    buffer.append(10)
    buffer.append(10)
    buffer.append(100)
    buffer.append(100)
    buffer.append(100)
    buffer.append(100)
    buffer.append(100)
    buffer.append(100)
    print(buffer[-10])