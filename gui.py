import os
from matplotlib import pyplot as plt
import cv2
import numpy as np


def get_roi_binary():
    timer = 0
    for i in os.listdir('temp'):
        img_path = os.path.join('temp',i)
        img = cv2.imread(img_path,0)
        while True:
            cv2.imshow('digi',img)
            k = cv2.waitKey(1)

            if k == ord('q'):
                break
            elif k == ord('s'):
                x,y,w,h = cv2.selectROI('digi',img,fromCenter=False)
                img_roi = img[y:y+h,x:x+w]
                img_roi = cv2.resize(img_roi,(50,90))
                break
        # print(np.unique(img_roi))
        # histogram, bin_edges = np.histogram(img_roi, bins=256, range=(0.0, 1.0))
        # fig, ax = plt.subplots()
        # plt.plot(bin_edges[0:-1], histogram)
        # plt.title("Grayscale Histogram")
        # plt.xlabel("grayscale value")
        # plt.ylabel("pixels")
        # plt.xlim(0, 255.0)
        # plt.show()

        ret2, img_roi_binary = cv2.threshold(img_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print(np.unique(img_roi_binary))
        cv2.imshow('roi', img_roi_binary)
        cv2.waitKey()
        save_name = os.path.join('temp','result_'+str(timer)+'.tiff')
        timer +=1
        cv2.imwrite(save_name,img_roi_binary)
        cv2.destroyAllWindows()
def refine_template():
    template_dir = 'OCR_template'
    if os.path.exists(template_dir):
        for i in range(10):
            img_file_path = os.path.join(template_dir, "result_" + str(i) + '.tiff')
            imfrag= cv2.imread(img_file_path, 0)
            _, imfrag_h = imfrag.shape
            ret2, imfrag = cv2.threshold(imfrag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
                # if h > 30 / 85 * imfrag_h:
                digitCnts.append(c)
                xloc = np.append(xloc, x)

            # if no connected component is detected, return ''
            if digitCnts == []:
                return ''
            # sort using x direction
            idx = np.argsort(xloc)
            tmp = digitCnts.copy()
            digitCnts = []
            for i in idx:
                digitCnts.append(tmp[i])

            digit = ''
            if len(digitCnts) > 3:
                print('detect error,Suggested click restart btn')
                return '-1'
            # print(len(digitCnts))
            for c in digitCnts:
                (x, y, w, h) = cv2.boundingRect(c)
                roi = imfrag[y:y + h, x:x + w]

                if roi is not None:
                    roi = cv2.resize(roi, (50, 90))
                    cv2.imshow('roi', roi)
                    cv2.waitKey()
                    cv2.imwrite(img_file_path,roi)

            print(f'[INFO] TEMPLATE {i} lOADED')
    else:
        raise FileNotFoundError('NO TEMPLATE')

if __name__ == '__main__':
    refine_template()
    get_roi_binary()
    camera_top = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    i =0
    while True:
        ret,frame = camera_top.read()
        if ret:
            # cv2.resizeWindow('breast', 640, 480)
            # cv2.resizeWindow('parameter', 640, 480)
            # 在窗口中展示画面
            cv2.imshow('breast', frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

            if k == ord('w'):
                breast_file_name = os.path.join('digi_' + str(i) + '.jpg')
                cv2.imwrite(breast_file_name, frame)
                i += 1
                print('saved')
        else:
            break

    camera_top.release()
    cv2.destroyAllWindows()