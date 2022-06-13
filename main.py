import asyncio
import os
import queue
import tkinter as tk
import threading
from datetime import datetime
import pickle
import cv2
import serial

from tof_functions import read_sensor
import numpy as np
from tools import order_points_new, crop_block, Buffer, block_analyse,model_infer

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
stop_queue = queue.Queue()

e = threading.Event()
org = (50, 50)
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.putText() method

def read_sensor(serialport='COM4', name='patient0'):
    print(serialport+"###starting thread###")
    data_list = []
    with serial.Serial(
            port=serialport,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=None) as ser:

        # data storage
        ser.reset_input_buffer()
        distarray = np.zeros((4, 4, 8))  # 4 sensors, 4 x 8 pixelmatrix
        points = np.zeros((4, 4, 8, 3))  # 4 sensors, 4 x 8 pixelmatrix, 3 coordinates (x, y, z)

        # debug_pose = set_debug_pose()

        while not e.is_set():  # <-- insert read flag here
            dataraw = bytearray(ser.read_until(b'\xff\xfa\xff\xfa'))
            data = dataraw[-44:]
            identifier = data[44 - 7]
            # print('Sensor ID : ',identifier)
            status = int.from_bytes(data[44 - 12:44 - 9], 'little')
            # print('Sensorstatus: ', status)
            if (data[44 - 8] == 1):
                for i in range(8):
                    distarray[identifier, 0, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                    # print("Reihe 1")
            elif (data[44 - 8] == 2):
                for i in range(8):
                    distarray[identifier, 1, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                # print("Reihe 2")
            elif (data[44 - 8] == 3):
                for i in range(8):
                    distarray[identifier, 2, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                # print("Reihe 3")
            elif (data[44 - 8] == 4):
                for i in range(8):
                    distarray[identifier, 3, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                    # print("Reihe 4")
            list1 = (['timestamp', datetime.now(), 'identifier', identifier, distarray[identifier, :, :]])
            data_list.append(list1)

        with open(name + '.pkl', 'wb') as f:
            pickle.dump(data_list, f)


class Record():
    def __init__(self,area_reader):
        self.area_reader=area_reader
        self.camera_top = None
        self.camera_bot = None
        self.stoped = False
        self.force = -1
        self.x = 0
        self.y = 0
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        codec2 = cv2.VideoWriter_fourcc(*'mp4v')
        self.patient_idx = 0
        self.pause_flag = True
        self.recoding_flag = False
        self.out_top_path = f'patient{self.patient_idx}_top.mp4'
        self.out_bot_path = f'patient{self.patient_idx}_bot.mp4'
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 10, (640, 480))
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec2, 10, (640, 480))

        self.frame = tk.Tk()  # frame 组件

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
        self.loop = asyncio.new_event_loop()
        self.loop2 = asyncio.new_event_loop()
        self.t_tof = threading.Thread(target=self.get_loop, args=(self.loop,))
        self.t_tof2 = threading.Thread(target=self.get_loop, args=(self.loop2,))
        self.t_tof.start()
        self.t_tof2.start()
        t = self.get_tof()
        t2 = self.get_tof2()
        asyncio.run_coroutine_threadsafe(t, self.loop)
        asyncio.run_coroutine_threadsafe(t2, self.loop2)
        self.buffer_froce = Buffer(50)

        self.frame.mainloop()

    def get_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    async def get_tof2(self):
        while True:
            try:
                stop_flag = stop_queue.get_nowait()
                if stop_flag:
                    break
            except queue.Empty:
                await asyncio.sleep(1)
            if self.recoding_flag:
                print('patient_' + str(self.patient_idx))
                read_sensor(serialport='COM5', name='patientb_' + str(self.patient_idx))
            else:
                print('tof is still waiting...')


    async def get_tof(self):
        while True:
            try:
                stop_flag = stop_queue.get_nowait()
                if stop_flag:
                    break
            except queue.Empty:
                await asyncio.sleep(1)
            if self.recoding_flag:
                print('patient_' + str(self.patient_idx))
                read_sensor(serialport='COM4', name='patient_' + str(self.patient_idx))
            else:
                print('tof is still waiting...')

    def start(self):
        self.camera_top = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_bot = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.job = self.frame.after(1, self.get_4_point())
        while self.stoped == False and self.camera_top.isOpened() and self.camera_bot.isOpened():
            self.job = self.frame.after(1, self.get_force())

        self.shut_down()

    def switch(self):
        self.stoped = True
        self.frame.after(1, self.shut_down())

    def new_writer(self):
        self.out_top_path = f'patient{self.patient_idx}_top.mp4'
        self.out_bot_path = f'patient{self.patient_idx}_bot.mp4'
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        codec2 = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec2, 25, (480, 640))

    def get_4_point(self):
        while True:
            ret2, frame_bot = self.camera_bot.read()
            frame_bot = np.rot90(frame_bot, 2)

            cv2.imshow('frame_bot', frame_bot)
            k = cv2.waitKey(1)

            if k == ord('q'):
                break
            elif k == ord('s'):
                x, y, w, h = cv2.selectROI('frame_bot', frame_bot, fromCenter=False)
                self.temp_turple = x, y, w, h
                break
        cv2.destroyAllWindows()

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
        frame_bot = np.rot90(frame_bot,2)
        ret, frame_top = self.camera_top.read()
        frame_bot_gray = cv2.cvtColor(frame_bot, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.temp_turple

        # dst = cv2.warpPerspective(frame_bot, self.M, (int(self.x), int(self.y))).astype(np.uint8)
        # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        # block1, block2, block3, block4 = crop_block(dst)
        block3 = frame_bot_gray[y:y + h, x:x + w]
        self.force = block_analyse(block3)
        self.buffer_froce.append(self.force)
        most = int(self.buffer_froce.most())
        if most > 5:
            self.recoding_flag = True
            e.clear()

        # self.force = np.random.randint(5, 300)
        if int(self.force) > 5 or most > 5:
            water_mark = 'Recording Is On!'
            self.my_label.config(text=water_mark)
            self.my_label.update()
            self.out_top.write(frame_top)
            self.out_bot.write(frame_bot)


        else:
            # cv2.destroyAllWindows()
            water_mark = 'Recording Is On!'

            self.my_label.config(text=water_mark)
            self.my_label.update()
            if self.recoding_flag:
                self.recoding_flag = False
                e.set()
                self.pause()
        breast_pred = self.area_reader.forward(cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)).astype(
            np.uint8)
        frame_top=cv2.putText(frame_top,water_mark, org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
        frame_bot=cv2.putText(frame_bot,water_mark, org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Recoding parameter', frame_bot)
        cv2.imshow('Recoding TOP', frame_top)
        cv2.imshow('Recoding TOP_breast_pred', breast_pred)
        cv2.waitKey(1)

    def pause(self):
        self.out_bot.release()
        self.out_top.release()
        self.patient_idx += 1
        self.new_writer()
        print(f'start on new Patient {self.patient_idx}')
        self.patient_label.config(text=f"Paitent : {self.patient_idx}")
        self.patient_label.update()
        return

    def shut_down(self):
        try:
            self.out_top.release()
            self.out_bot.release()
            stop_queue.put(True)
            stop_queue.put(True)
            e.set()
        except AttributeError:
            print('Camera not start')

        # self.out_bot.release()

        if self.stoped == False:

            pass
        else:
            try:
                self.camera_top.release()
                self.camera_bot.release()
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
            self.t_tof2.join()
            self.frame.after(1, self.frame.quit)


if __name__ == '__main__':
    area_reader = model_infer(
        r'.\GoodModel\res34epoch=191-val_Iou=0.78.ckpt')
    R = area_reader()
