import numpy as np

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


