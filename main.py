import asyncio
import os
import queue
import time
import tkinter as tk
import threading
import cv2
import numpy as np
from tools import order_points_new, crop_block


# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
q  = queue.Queue()
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
        self.pause_flag = True
        self.recoding_flag = False
        self.out_top_path = f'patient{self.patient_idx}_top.mp4'
        self.out_bot_path = f'patient{self.patient_idx}_bot.mp4'
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        self.out_bot = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        self.template_dir = 'OCR_template'
        self.img_template = []
        if os.path.exists(self.template_dir):
            for i in range(10):
                img_file_path = os.path.join(self.template_dir, "result_" + str(i) + '.tiff')
                self.img_template.append(cv2.imread(img_file_path, 0))
                print(f'[INFO] TEMPLATE {i} lOADED')
        else:
            raise FileNotFoundError('NO TEMPLATE')

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
        self.temp_turple = ()
        self.loop =  asyncio.new_event_loop()
        self.t_tof= threading.Thread(target=self.get_loop,args=(self.loop ,    ))
        self.t_tof.start()
        t = self.get_tof()
        asyncio.run_coroutine_threadsafe(t, self.loop)
        self.frame.mainloop()

    def get_loop(self,loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def get_tof(self):
        while True:

            try:
                c = q.get()
                if not c:
                    break
            except:
                print('jo')
                print(c)



    def start(self):
        self.camera_top = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_bot = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.job = self.frame.after(1,self.get_4_point())
        while self.stoped == False and self.camera_top.isOpened():
            # self.get_force()
            self.job = self.frame.after(1, self.get_force())

        self.shut_down()

    def switch(self):
        self.stoped = True
        self.frame.after(1,self.shut_down())

    def new_writer(self):
        self.out_top_path = f'patient{self.patient_idx}_top.mp4'
        # self.out_bot_path = f'patient{self.patient_idx}_bot.mp4'
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        # self.out_bot = cv2.VideoWriter(self.out_bot_path, codec, 10, (1080, 1920))

    def get_4_point(self):
        while True:
            ret2, frame_bot = self.camera_bot.read()

            cv2.imshow('frame_bot', frame_bot)
            k = cv2.waitKey(1)

            if k == ord('q'):
                break
            elif k == ord('s'):
                x, y, w, h = cv2.selectROI('frame_bot', frame_bot, fromCenter=False)
                self.temp_turple = x, y, w, h
                break


    # def start_cam(self):
    #     self.camera_top = cv2.VideoCapture(0)
    #     # self.camera_bot = cv2.VideoCapture(1)
    #     # return self.camera_bot.isOpened() and self.camera_top.isOpened()
    #     return self.camera_top.isOpened()
    #     # return True

    def get_M_bot(self, fixed_4points):

        pts1, self.x, self.y = order_points_new(fixed_4points)
        pts2 = np.float32([[0, 0], [self.x, 0], [0, self.y], [self.x, self.y]])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)

    def get_force(self):
        ret2, frame_bot = self.camera_bot.read()
        ret, frame_top = self.camera_top.read()
        frame_bot = cv2.cvtColor(frame_bot, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.temp_turple
        # dst = cv2.warpPerspective(frame_bot, self.M, (int(self.x), int(self.y))).astype(np.uint8)
        # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        # block1, block2, block3, block4 = crop_block(dst)
        block3 = frame_bot[y:y + h, x:x + w]

        self.force = self.block_analyse(block3)


        # self.force = np.random.randint(5, 300)
        if self.force > 5:
            self.my_label.config(text='Recording Is On!')
            self.my_label.update()
            self.out_top.write(frame_top)
            self.recoding_flag =True
            cv2.imshow('Recoding parameter', frame_bot)
            cv2.waitKey(1)
            cv2.imshow('Recoding TOP',frame_top)
            cv2.waitKey(1)
            return
            # self.job = self.frame.after(1, self.get_force())

            # self.out_bot.write(frame_bot)
        else:
            cv2.destroyAllWindows()
            self.my_label.config(text='Recording Is off!')
            self.my_label.update()
            if self.recoding_flag :
                self.recoding_flag=False
                self.pause()
            return

    def pause(self):
        self.out_top.release()
        self.patient_idx += 1
        self.new_writer()
        print(f'start on new Patient {self.patient_idx}')
        self.patient_label.config(text=f"Paitent : {self.patient_idx}")
        self.patient_label.update()
        return
    def block_analyse(self, imfrag):
        # new method of reading digits in the imfrag
        _, imfrag_h = imfrag.shape
        ret2, imfrag = cv2.threshold(imfrag, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
        if len(digitCnts) != 1:
            print('detect error,Suggested click restart btn')
            return -1
        # print(len(digitCnts))
        for c in digitCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = imfrag[y:y + h, x :x + w]

            if roi is not None:
                roi = cv2.resize(roi, (50, 90))
                # cv2.imshow('roi',roi)
                # cv2.waitKey()
                roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

                score = np.zeros(10)
                for i in range(10):
                    score[i] = self.get_match_score(roi, self.img_template[i])
                digit += str(np.argmax(score))
            else:
                digit = 0
        print(digit)
        return int(digit)

    def get_match_score(self, img, template):
        # print(np.max(template))
        tp = (img == 255) == (template == 255)
        fp = (img == 0) == (template == 0)
        tn = (img == 255) == (template == 0)
        fn = (img == 0) == (template == 255)

        score = np.sum(tp) + np.sum(fp) - np.sum(tn) - np.sum(fn)
        return score

    def shut_down(self):
        try:
            self.out_top.release()
            q.put(False)
        except AttributeError:
            print('Camera not start')

        # self.out_bot.release()

        if self.stoped == False:

            pass
        else:
            try:
                self.camera_top.release()
            except AttributeError:
                print('Camera not start')
            # self.camera_bot.release()
            print('Done')
            cv2.destroyAllWindows()
            print(self.out_top_path)
            # self.loop.stop()
            # self.loop.close()
            self.loop.call_soon_threadsafe(self.loop.stop)

            self.t_tof.join()
            self.frame.after(1,self.frame.quit)


if __name__ == '__main__':
    R = Record()
