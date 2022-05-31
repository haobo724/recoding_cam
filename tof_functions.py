import serial
import numpy as np
from datetime import datetime
import pickle as p


# returns pose (= orientation + position) for one sensor (4x4 matrix)
def get_sens_pose_single(sensXPos, sensYPos, sensZPos, sensXAngle, sensZAngle):
    radX = np.radians(sensXAngle)
    radZ = np.radians(sensZAngle)
    senspose = np.array([[np.cos(radZ), -np.sin(radZ) * np.cos(radX), -np.sin(radZ) * np.sin(radX), 0],
                         [np.sin(radZ), np.cos(radZ) * np.cos(radX), -np.cos(radZ) * np.sin(radX), 0],
                         [0, np.sin(radX), np.cos(radX), 0],
                         [0, 0, 0, 1]])
    vect = np.array([sensXPos, sensYPos, sensZPos, 1])
    senspose[:, 3] = vect
    return senspose


# returns pose (= orientation + position) for multiple sensors (4x4 matrix) compared to base coordinate system
def get_sens_pose_multiple(sensXPos, sensYPos, sensZPos, sensXAngle, sensZAngle):
    if (len(sensXPos) == len(sensYPos) == len(sensZPos) == len(sensXAngle) == len(sensZAngle)):
        senspose = np.zeros([len(sensXPos), 4, 4])
        for i in range(len(sensXPos)):
            radX = np.radians(sensXAngle[i])
            radZ = np.radians(sensZAngle[i])
            senspose[i, :, :] = np.array([[np.cos(radZ), -np.sin(radZ) * np.cos(radX), -np.sin(radZ) * np.sin(radX), 0],
                                          [np.sin(radZ), np.cos(radZ) * np.cos(radX), -np.cos(radZ) * np.sin(radX), 0],
                                          [0, np.sin(radX), np.cos(radX), 0],
                                          [0, 0, 0, 1]])
            vect = np.array([sensXPos[i], sensYPos[i], sensZPos[i], 1])
            senspose[i, :, 3] = vect
        return senspose
    else:
        return 0


# transforms one point to the base coordinate system
def point_coordinate_transform_single(point, senspose):
    if (point.shape == (3,) or point.shape == (1, 3)):
        transformed_point = np.matmul(senspose[0:3, 0:3], point) + senspose[0:3, 3]
        return transformed_point
    else:
        print("Error: wrong point shape")
        return 0


# transforms multiple points to the base coordinate system
def point_coordinate_transform_multiple(points, sensposes):
    if (len(points) == len(sensposes)):
        transformed_points = np.zeros([len(points), 4, 8, 3])
        for i in range(len(points)):
            for j in range(4):
                for b in range(8):
                    transformed_points[i, j, b, :] = np.matmul(sensposes[i, 0:3, 0:3], points[i, j, b, :]) + sensposes[
                                                                                                             i, 0:3, 3]
        return transformed_points
    else:
        print("Error: Input wrong dimensions, first dimension of points and sensposes must be equal");


def sens_dist_to_point_single(distarray):  # distarray = 4x8 matrix
    if (len(distarray[:, 0]) == 4 and len(distarray[0, :]) == 8):
        fov_hor = 12.4
        fov_ver = 5.4
        anglearray_hor = np.zeros([4, 8])
        anglearray_ver = np.zeros([4, 8])
        for i in range(4):
            anglearray_hor[i, :] = -np.radians(np.linspace(fov_hor / 16, fov_hor - fov_hor / 16, 8) - fov_hor / 2)
        for i in range(8):
            anglearray_ver[:, i] = -np.radians(np.linspace(fov_ver / 8, fov_ver - fov_ver / 8, 4) - fov_ver / 2)
        print(anglearray_hor)
        points = np.zeros([4, 8, 3])  # 4x8 matrix with 3 layers (x y z position)
        for i in range(4):
            for j in range(8):
                points[i, j, 0] = np.sin(anglearray_hor[i, j]) * distarray[i, j]
                points[i, j, 1] = np.cos(anglearray_hor[i, j]) * distarray[i, j]
                points[i, j, 2] = np.sin(anglearray_ver[i, j]) * distarray[i, j]
        return points
    else:
        print("Error: Input wrong dimensions, must be 4x8 matrix");


def sens_dist_to_point_multiple(distarray):  # distarray = 4x8 matrix
    if (1):
        fov_hor = 12.4
        fov_ver = 5.4
        anglearray_hor = np.zeros([4, 8])
        anglearray_ver = np.zeros([4, 8])
        for i in range(4):
            # anglearray_hor[i,:] = np.radians(np.linspace(fov_hor/16, fov_hor-fov_hor/16,8) - fov_hor/2)
            anglearray_hor[i, :] = np.radians(np.linspace(fov_hor / 16, fov_hor - fov_hor / 16, 8) - fov_hor / 2 - 2.7)
        for i in range(8):
            # anglearray_ver[:,i] = np.radians(np.linspace(fov_ver/8, fov_ver-fov_ver/8,4)-fov_ver/2)
            anglearray_ver[:, i] = - np.radians(np.linspace(fov_ver / 8, fov_ver - fov_ver / 8, 4) - fov_ver / 2)
        # print(anglearray_hor)
        points = np.zeros([len(distarray), 4, 8, 3])  # multiple of 4x8 matrix with 3 layers (x y z position)
        for b in range(len(distarray)):
            for i in range(4):
                for j in range(8):
                    points[b, i, j, 0] = np.sin(anglearray_hor[i, j]) * distarray[b, i, j]
                    points[b, i, j, 1] = np.cos(anglearray_hor[i, j]) * distarray[b, i, j]
                    points[b, i, j, 2] = np.sin(anglearray_ver[i, j]) * distarray[b, i, j]
        return points
    else:
        print("Error: Input wrong dimensions, must be 4x8 matrix");


def set_debug_pose():
    # generating 4 poses for 3d printed sensor mount
    sensXPos = np.array([0, -30, 0, -30])
    sensYPos = np.array([0, 0, 0, 0])
    sensZPos = np.array([19, 19, 55, 55])
    sensXAngle = np.array([0, 0, 0, 0])
    sensZAngle = np.array([13, 25, 13, 25])
    senspose = get_sens_pose_multiple(sensXPos, sensYPos, sensZPos, sensXAngle, sensZAngle)
    return senspose


def read_sensor(serialport='COM4', data_list=[]):
    print("###starting thread###")
    with serial.Serial(
            port=serialport, \
            baudrate=115200,
            parity=serial.PARITY_NONE, \
            stopbits=serial.STOPBITS_ONE, \
            bytesize=serial.EIGHTBITS, \
            timeout=None) as ser:

        # data storage
        distarray = np.zeros((4, 4, 8))  # 4 sensors, 4 x 8 pixelmatrix
        points = np.zeros((4, 4, 8, 3))  # 4 sensors, 4 x 8 pixelmatrix, 3 coordinates (x, y, z)

        debug_pose = set_debug_pose()

        while (1):  # <-- insert read flag here
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
            print(distarray)
        with open("save_list.p", 'wb') as f:
            f.dump(data_list)

if __name__ == '__main__':
    read_sensor()