import os
import tkinter as tk

import cv2
import numpy as np
from tools import order_points_new, crop_block


# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class Record():
    def __init__(self):
        self.camera_top = None
        self.camera_bot = None
        self.stoped = False
        self.force = -1
        self.x = 0
        self.y = 0
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.patient_idx = 0
        self.out_top_path = f'patient{self.patient_idx}_top.mp4'
        self.out_bot_path = f'patient{self.patient_idx}_bot.mp4'
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 10, (1080, 1920))
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec, 10, (1080, 1920))
        self.template_dir = 'OCR_template'
        self.img_template = []
        if os.path.exists(self.template_dir):
            for i in range(10):
                img_file_path = os.path.join(self.template_dir, str(i) + '.jpg')
                self.img_template.append(cv2.imread(img_file_path, 0))
        else:
            pass
            # raise FileNotFoundError('NO TEMPLATE')

        self.frame = tk.Tk()  # frame 组件

        # frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.stoped_b = tk.Button(self.frame, text='stoped', bg='black', fg='white',
                                         command=self.switch)  # Button按钮, command中调用定义的方法
        self.my_label = tk.Label(self.frame,

                                 text="Recording Is off! ",

                                 fg="green",

                                 font=("Helvetica", 32))
        self.patient_label = tk.Label(self.frame,

                                      text=f"Paitent : {self.patient_idx}",

                                      fg="red",

                                      font=("Helvetica", 12))
        self.my_label.pack(pady=20)
        self.patient_label.pack(pady=10)
        self.stoped_b.pack()
        self.start_b = tk.Button(self.frame, text='start', bg='black', fg='white',
                                  command=self.start)  # Button按钮, command中调用定义的方法
        self.start_b.pack()

        # self.get_force()
        self.frame.mainloop()

    def pipeline(self):
        if self.start_cam():
            # R.get_M_bot([1, 2, 3, 4])
            while not self.stoped:
                self.get_force()
    def start(self):
        while self.stoped==False:
            self.job =self.frame.after(1, self.get_force())
    def switch(self):
        self.stoped = True
        self.shut_down()



    def new_writer(self):
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 10, [1080, 1920])
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec, 10, [1080, 1920])

    def __str__(self):
        return

    def start_cam(self):
        # self.camera_top = cv2.VideoCapture(0)
        # self.camera_bot = cv2.VideoCapture(1)
        # return self.camera_bot.isOpened() and self.camera_top.isOpened()
        return True

    def get_M_bot(self, fixed_4points):
        pts1, self.x, self.y = order_points_new(fixed_4points)
        pts2 = np.float32([[0, 0], [self.x, 0], [0, self.y], [self.x, self.y]])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)

    def get_force(self):
        # ret2, frame_bot = self.camera_bot.read()
        # ret, frame_top = self.camera_top.read()
        #
        # dst = cv2.warpPerspective(frame_bot, self.M, (int(self.x), int(self.y))).astype(np.uint8)
        # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        # block1, block2, block3, block4 = crop_block(dst)
        # self.force = self.block_analyse(block3)
        self.force = np.random.randint(0,20)
        if self.force > 5:
            self.my_label.config(text='Recording Is On!')
            return
            # self.job = self.frame.after(1, self.get_force())

            # self.out_bot.write(frame_bot)
            # self.out_top.write(frame_top)
        else:
            self.my_label.config(text='Recording Is off!')
            self.shut_down()
            return


    def block_analyse(self, imfrag):
        # new method of reading digits in the imfrag
        _, imfrag_h = imfrag.shape

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

    def shut_down(self):
        # self.out_top.release()
        # self.out_bot.release()

        if  self.stoped == False:
            self.patient_idx += 1
            # self.new_writer()
            print(f'start on new Patient {self.patient_idx}')
            self.patient_label.config(text=f"Paitent : {self.patient_idx}")
            self.patient_label.update()

            self.get_force()
        else:
            # self.camera_top.release()
            # self.camera_bot.release()
            print('Done')
            self.frame.quit()



if __name__ == '__main__':
    R = Record()

