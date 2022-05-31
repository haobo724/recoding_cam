import queue

import numpy as np

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