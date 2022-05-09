"""
Util to recognize the digits in the param image

"""
import glob
import os
import pickle
import re
import time
from paddleocr import PaddleOCR,draw_ocr
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

'''
:return: [angle, height, compression force, name(void)]
'''


class digit_reader:
    def __init__(self, img_saveEvent=True, use_new_ocr=True):
        self.threshold_img = None
        self.adaptive_thresh_img = None
        self.img_saveEvent = img_saveEvent
        self.text_info = None
        # for simple ocr
        self.angel_block = None
        self.distance_block = None
        self.press_block = None
        self.Name_block = None
        self.use_new_ocr = use_new_ocr
        self.easyOcr = True
        self.double_check = True
        if self.easyOcr:
            self.OCR_reader = easyocr.Reader(['en', 'de'], gpu=True)

        # warp mtx
        self.M = None
        self.warp_size = []
        # if os.path.exists('M.pkl'):
        #     with open('M.pkl', 'rb') as m:
        #         self.M,self.warp_size = pickle.load(m)

        # digit template INIT
        self.img_template = []
        self.root_dir = os.path.dirname(os.path.dirname(__file__))
        self.save_path = os.path.join(self.root_dir, 'Template_saved')

        self.template_dir = os.path.join(self.root_dir, 'OCR_template')
        if os.path.exists(self.template_dir):
            for i in range(10):
                img_file_path = os.path.join(self.template_dir, str(i)+ '.jpg')
                self.img_template.append(cv2.imread(img_file_path, 0))
        else:
            raise FileNotFoundError('NO TEMPLATE')

    def imread(self, img_org):
        # warp image and OCR
        if self.M is None:
            print('start ini')
            img = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
            rows, cols = img.shape

            # do thresholding to the image
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.threshold_img = thresh
            pts1, pts2, x, y = self.get_corner_4points_C(thresh)

            # pts1, x, y = self.get_point_song(thresh)

            # pts1 = np.float32([[c1, r1], [c2, r2], [c3, r3], [c4, r4]])

            pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])

            # calibrate the distorted display rectangle using function of opencv
            self.M = cv2.getPerspectiveTransform(pts1, pts2)
            self.warp_size = [x, y]

        else:
            x = self.warp_size[0]
            y = self.warp_size[1]

        self.crop_img = cv2.warpPerspective(img_org.astype(np.uint8), self.M, (int(x), int(y)))


        # Block segmentation
        img = cv2.cvtColor(self.crop_img, cv2.COLOR_RGB2GRAY)
        y, x = img.shape

        # try equalizeHist method to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(21, 21))
        img = clahe.apply(img)

        # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        self.adaptive_thresh_img = thresh

        cv2.imwrite(os.path.join(self.save_path, "calibration_thresh.jpg"), thresh)
        block1, block2, block3, block4 = self.crop_block(thresh, x, y)

        self.angel_block = block1
        self.distance_block = block2
        self.press_block = block3
        self.Name_block = block4

        # TODO: READ NAME IS IT VALUABLE
        if self.easyOcr:
            Name = self.new_detect(Only_txt=True)
        else:
            Name = 'disabled'
        cv2.imwrite(os.path.join(self.save_path,'block1.jpg'), self.angel_block)
        cv2.imwrite(os.path.join(self.save_path,'block2.jpg'), self.distance_block)
        cv2.imwrite(os.path.join(self.save_path,'block3.jpg'), self.press_block)

        code_template = [self.block_analyse(self.angel_block),
                         self.block_analyse(self.distance_block),
                         self.block_analyse(self.press_block),
                         Name]
        # print(code_template)
        return code_template

    def imread_given_point(self, img_org,pts1=np.float32([[92, 220], [473, 193], [71, 454], [527, 427]])):
        # warp image and OCR
        img = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
        rows, cols = img.shape
        pts1, x, y = self.order_points_new(pts1)
        pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img_org.astype(np.uint8), self.M, (int(x), int(y))).astype(np.uint8)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        self.crop_img = dst

        # do thresholding to the image
        ret, thresh = cv2.threshold(self.crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        self.threshold_img = thresh

        y, x = thresh.shape

        self.adaptive_thresh_img = thresh
        cv2.imwrite(os.path.join(self.save_path, "calibration_thresh.jpg"), thresh)

        block1, block2, block3, block4 = self.crop_block(thresh, x, y)

        self.angel_block = block1
        self.distance_block = block2
        self.press_block = block3
        self.Name_block = block4

        # TODO: READ NAME IS IT VALUABLE
        if self.easyOcr:
            Name = self.new_detect(Only_txt=True)
        else:
            Name = 'disabled'
        cv2.imwrite(os.path.join(self.save_path,'block1.jpg'), self.angel_block)
        cv2.imwrite(os.path.join(self.save_path,'block2.jpg'), self.distance_block)
        cv2.imwrite(os.path.join(self.save_path,'block3.jpg'), self.press_block)
        code_template = [self.block_analyse(self.angel_block),
                         self.block_analyse(self.distance_block),
                         self.block_analyse(self.press_block),
                         Name]
        # print(code_template)
        return code_template

    def get_corner_4points_C(self, afterThresh):
        rows, cols = afterThresh.shape

        xv, yv = np.meshgrid(np.arange(afterThresh.shape[1]), np.arange(afterThresh.shape[0]))

        dst1 = xv + yv
        dst2 = xv - yv
        dst3 = -dst1
        dst4 = -dst2
        dst1[afterThresh == 0] = np.amax(dst1)

        dst2[afterThresh == 0] = np.amax(dst2)
        dst3[afterThresh == 0] = np.amax(dst3)
        dst4[afterThresh == 0] = np.amax(dst4)

        r1, c1 = np.where(dst1 == np.amin(dst1))
        r3, c3 = np.where(dst2 == np.amin(dst2))
        r4, c4 = np.where(dst3 == np.amin(dst3))
        r2, c2 = np.where(dst4 == np.amin(dst4))

        r1 = r1[0]
        c1 = c1[0]
        r2 = r2[0]
        c2 = c2[0]
        r3 = r3[0]
        c3 = c3[0]
        r4 = r4[0]
        c4 = c4[0]

        x = c2 - c1
        y = max([r3, r4]) - min([r1, r2]) + int(1 / 10 * rows)
        pts1 = np.float32([[c1, r1], [c2, r2], [c3, r3], [c4, r4]])
        pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
        return pts1, pts2, x, y

    def get_warp_mtx(self, img):
        # warp image and OCR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(os.path.join(self.save_path, 'original.jpg'), img)

        # do thresholding to the image
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.threshold_img = thresh
        cv2.imwrite(os.path.join(self.save_path, 'threshold_img.jpg'), thresh)

        # finding all contours of the images
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)

        # finding the biggest contour (ROI)
        nb_components, output1, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1

        # if len(sizes) < 2:
        #     return code

        max_size = sizes[1]
        for ii in range(2, nb_components):
            if sizes[ii] > max_size:
                max_label = ii
                max_size = sizes[ii]
        img2 = np.zeros(output1.shape).astype('uint8')
        img2[output1 == max_label] = 255

        # only keep the biggest contour
        afterThresh = np.zeros_like(img)

        afterThresh[img2 == 255] = thresh[img2 == 255]
        cv2.imwrite(os.path.join(self.save_path, 'afterThresh.jpg'), afterThresh)

        pts1, pts2, x, y = self.get_corner_4points_C(afterThresh)
        # calibrate the distorted display rectangle using function of opencv
        self.M = cv2.getPerspectiveTransform(pts1, pts2)

        self.warp_size = [x, y]
        with open("M.pkl", 'wb') as f:
            pickle.dump((self.M, self.warp_size), f)

    def order_points_new(self, pts):
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

    def delet_contours(self, contours, delete_list):
        # delta作用是offset，因为del是直接pop出去，修改长度了
        delta = 0
        for i in range(len(delete_list)):
            # print("i= ", i)
            del contours[delete_list[i] - delta]
            delta = delta + 1
        return contours

    def get_point_song(self, thresh):
        # finding all contours of the images
        code = ['', '', '', '']

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        delete_list = []
        for i in range(len(contours)):
            if (cv2.contourArea(contours[i]) < 500):
                delete_list.append(i)
        # delet contour 是序号
        contours = self.delet_contours(contours, delete_list)
        contours = sorted(contours, key=cv2.contourArea)[-1]
        temp = np.zeros_like(thresh)
        if len(contours) == 0:
            return code
        thresh = cv2.fillPoly(temp, [contours], (255, 255, 255))

        afterThresh = thresh.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
        afterThresh = cv2.morphologyEx(afterThresh, cv2.MORPH_OPEN, kernel, iterations=5)  # 开运算1

        edged = cv2.Canny(afterThresh, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        contours = np.array(contours).squeeze(1)
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = contours.sum(axis=1)
        rect[0] = contours[np.argmin(s)]
        rect[2] = contours[np.argmax(s)]
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(contours, axis=1)
        rect[1] = contours[np.argmin(diff)]
        rect[3] = contours[np.argmax(diff)]
        afterThresh = np.dstack((afterThresh, afterThresh, afterThresh))
        pts1 = []
        for pt in rect:
            b = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            r = np.random.randint(0, 256)
            x = int(pt[0])
            y = int(pt[1])
            cv2.circle(afterThresh, (x, y), 5, (int(b), int(g), int(r)), 2)
            pts1.append([x, y])
        # self.temp_show(afterThresh)

        pts1, x, y = self.order_points_new(np.float32(pts1))
        return pts1, x, y

    def new_detect(self, Only_txt=False):
        if Only_txt:
            # self.temp_show(self.Name_block)

            results4 = self.OCR_reader.recognize(self.Name_block, detail=0,
                                                 paragraph=False)
            if len(results4) > 2:
                for result in results4:
                    if len(result) > 2:
                        txt = re.findall(r'[a-zA-Z]', result)
                        results4 = txt
                        break
            else:
                results4 = re.findall(r'[a-zA-Z]', results4[0])
            return results4

        # self.temp_show(self.angel_block)
        results1 = self.OCR_reader.readtext(self.angel_block, detail=0,
                                            paragraph=False)[0]  # Set detail to 0 for simple text output
        # self.temp_show(self.distance_block)

        results2 = self.OCR_reader.readtext(self.distance_block, detail=0,
                                            paragraph=False)[0]  # Set detail to 0 for simple text output
        # self.temp_show(self.press_block)

        results3 = self.OCR_reader.readtext(self.press_block, detail=0,
                                            paragraph=False)[0]  # Set detail to 0 for simple text output
        # self.temp_show(self.Name_block)
        results4 = self.OCR_reader.readtext(self.Name_block, detail=0,
                                            paragraph=False)

        return [results1, results2, results3, results4]

    def crop_block(self, thresh, x, y):
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

    def temp_show(self, img):
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.show()

    def block_analyse(self, imfrag):
        # new method of reading digits in the imfrag
        _, imfrag_h = imfrag.shape

        if imfrag is None:
            imfrag = self.press_block
        # detect single digit and detect
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素

        # convert gray value for contour detection
        cnts, _ = cv2.findContours(imfrag.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # default using opencv3, if opencv2 is used, here use cnts[0]
        # cnts = cnts[0]
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
            roi = imfrag[y - 2:y + h + 2, x - 2:x + w + 2]

            if roi is not None:
                try:
                    roi = cv2.resize(roi, (40, 80))
                except:
                    print('something wrong')
                    roi = np.zeros((40, 80))
                roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

                score = np.zeros(10)
                for i in range(10):
                    score[i] = self.get_match_score(roi, self.img_template[i])
                digit += str(np.argmax(score))
            else:
                digit = 0

        return digit

    def get_match_score(self, img, template):
        # print(np.max(template))
        tp = (img == 255) == (template == 255)
        fp = (img == 0) == (template == 0)
        tn = (img == 255) == (template == 0)
        fn = (img == 0) == (template == 255)

        score = np.sum(tp) + np.sum(fp) - np.sum(tn) - np.sum(fn)
        return score


def video_read():
    cap = cv2.VideoCapture(r'F:\Siemens\Pressure_measure_activate_tf1x\Detection_util\para.avi')
    reader = digit_reader()
    #
    # cv2.namedWindow("image")
    # cv2.setMouseCallback("image", mouse_click_callback)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # cv2.setMouseCallback("image", mouse_click_callback)

            param = reader.imread_given_point(frame)
            print('param', param)
            cv2.putText(frame, param[1], (50, 150), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 12)
            cv2.imshow('image', frame)

            k = cv2.waitKey(5)
            # q键退出
            if (k & 0xff == ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()


def mouse_click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def test_OCR():
    reader=digit_reader()
    paths = glob.glob(r'.\Test\parameter\*.jpg')
    for i in range(len(paths)):
        reader.imread(cv2.imread(paths[i]))

def time_costum(func):
    start = time.time()
    def wrapper(*args,**kwargs):
        func(*args, **kwargs)
        print(time.time() - start)
        return
    return wrapper

@time_costum
def test_paddleOCR():
    img_path= r'../Demo/demo_parameter.JPG'

    # img=cv2.imread(img_path,0)
    ocr =PaddleOCR(use_angle_cls=True,lang='en')
    result = ocr.ocr(img_path,cls=True)
    for line in result:
        print(line)
    # draw result
    # from PIL import Image
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')

if __name__ == '__main__':
    test_paddleOCR()
    # Run the main function for testing, this will create some images and files in current directory.
    # paths = glob.glob(r'C:\ChangLiu\Raspberry_proj\Pi_img\Pi_img\dark\*.jpg')
    # paths = glob.glob(r'C:\Users\z00461wk\Desktop\Pressure_measure_activate_tf1x\para\IMG_4278.JPG')
    # print(paths)
    # paths = [r'F:\Siemens\Pressure_measure_activate_tf1x\test/param.JPG']
    # paths =  glob.glob(r'F:\Siemens\Pressure_measure_activate_tf1x\Camera_util\new_para\10\*.jpg')
    # resize_test(paths)
    # resize_test(paths)
