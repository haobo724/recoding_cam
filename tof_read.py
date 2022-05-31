import pickle

import serial


infile = open('patient_0.pkl', 'rb')
# load calibration parameters
tof = pickle.load(infile)
infile.close()

print(tof[0])
print(tof[4])