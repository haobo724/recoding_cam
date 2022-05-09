import tkinter as tk
class gui():
    def __init__(self):
        self.frame = tk.Tk()  # frame 组件
        self.patient_idx = 0
        # frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.stoped = False

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

    def start(self):
        pass
    def switch(self):
        pass
    def change_idx(self):
        self.patient_idx +=1
