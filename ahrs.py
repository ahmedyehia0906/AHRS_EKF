import math
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    columns = [df[col].tolist() for col in df.columns]
    return columns

def average(list):
    return sum(list)/len(list)

def calibrate(list, bias):
    
    result = []
    for x in list:
        result.append(x - bias)
    
    return result


def calibrate_magnetometer(raw_data, calibration_matrix):
    """
    Calibrates magnetometer data using a calibration matrix.

     """
    return np.dot(raw_data, calibration_matrix.T)

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


'''
def earth_frame_conversion(acceleration, magnetic_field, attitude):
    # Normalize accelerometer and magnetometer readings
    accel_norm = normalize(acceleration)
    mag_norm = normalize(magnetic_field)

    # Convert accelerometer reading to the Earth frame
    gravity = np.array([0, 0, 1])  # Gravity vector in the Earth frame
    earth_accel = np.dot(attitude, accel_norm)  # Convert to Earth frame

    magnetic_field_earth = mag_norm - np.dot(mag_norm, gravity) * gravity


    heading = np.arctan2(magnetic_field_earth[1], magnetic_field_earth[0])

    return earth_accel, heading
  '''
  
def earth_frame_conversion(acceleration, magnetic_field):
    
    # Normalize accelerometer and magnetometer readings
    accel_norm = normalize(acceleration)
    mag_norm = normalize(magnetic_field)

    # Compute the rotation axis as the cross product of gravity and magnetic field
    rotation_axis = np.cross(accel_norm, mag_norm)
    
    # Normalize the rotation axis
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Compute the rotation angle using the dot product
    rotation_angle = np.arccos(np.dot(accel_norm, mag_norm))

    # Compute the attitude matrix using the axis-angle representation
    attitude_matrix = axis_angle_to_rotation_matrix(rotation_axis, rotation_angle)

    return attitude_matrix

def axis_angle_to_rotation_matrix(axis, angle):
    
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    rotation_matrix = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])
    return rotation_matrix


def rotationX(angle):
    
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)

    rotation_matrixX = np.array([
        [1, 0, 0],
        [0, cos_val, -sin_val],
        [0, sin_val, cos_val]
    ])

    return rotation_matrixX
def rotationY(angle):
    
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    
    rotation_matrixY = np.array([[cos_val, 0, sin_val]
                                 ,[0, 1, 0],
                                 [-sin_val, 0, cos_val]])
    return rotation_matrixY

def rotationZ(angle):
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    rotation_matrixZ = np.array([[cos_val, -sin_val, 0],
                                 [sin_val, cos_val,0]
                                 [0, 0, 1]])
    return rotation_matrixZ

def transformation_matrix(rotation_matrix, vector):
    
    if rotation_matrix.shape != (3, 3) or vector.shape != (3, 1):
        raise ValueError("Invalid matrix dimensions.")

   
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = vector.flatten()

    return transformation_matrix


# Loading the data
column_lists_cal = read_csv_file('Stationary_laying_1.csv')
magneteometer_Calibration_data = read_csv_file('Magnetometer calibration.csv')
data = read_csv_file('data_120.csv')

# data for calibration
Ax = column_lists_cal[7]
Ay = column_lists_cal[8]
Az = column_lists_cal[9]
Gx = column_lists_cal[10]
Gy = column_lists_cal[11]
Gz = column_lists_cal[12]
Mx = np.array(magneteometer_Calibration_data[13])
My = np.array(magneteometer_Calibration_data[14])
Mz = np.array(magneteometer_Calibration_data[15])


# averaging to calibrate
avg_Ax = average(Ax)
avg_Ay = average(Ay)
avg_Az = average(Az)
avg_Gx = average(Gx)
avg_Gy = average(Gy)
avg_Gz = average(Gz)

# accel and gyro and magn data
ticks,ax,ay,az,gx,gy,gz,mx,my,mz = data[0],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13]

# calibration
ax_calibrated = calibrate(ax, avg_Ax)
ay_calibrated = calibrate(ay, avg_Ay)
az_calibrated = calibrate(az, avg_Az)
gx_calibrated = calibrate(gx, avg_Gx)
gy_calibrated = calibrate(gy, avg_Gy)
gz_calibrated = calibrate(gz, avg_Gz)

# Magnetometer calibration
magnetometer_calibration_txtPath = 'mag_cal.txt'
np.savetxt(magnetometer_calibration_txtPath, np.column_stack((Mx,My,Mz)))
magnetometer_bias = np.array([-0.383789, -0.03366, -0.024828])
manetometer_transform = np.array([[12.291285,0.530290,0.011556],
                                 [0.530290,12.451910,0.040250],
                                 [0.011556,0.040250,12.378532]])
magnetometer_IMUF = np.column_stack((mx, my, mz))
print(magnetometer_IMUF[0])
#subtract the bias
magnetometer_IMUF_cal = magnetometer_IMUF - magnetometer_bias
magnetometer_IMUF_cal = calibrate_magnetometer(magnetometer_IMUF_cal,manetometer_transform)
print(magnetometer_IMUF_cal[0])
#print(np.dot(manetometer_transform,np.transpose(magnetometer_IMUF[0])))


#df1 = pd.DataFrame({'x': ticks[0:100], 'y': gx_calibrated[0:100]})
#df2 = pd.DataFrame({'x': ticks[0:100], 'y': gx[0:100]})
#ax = df1.plot(x='x', y='y', kind='line', marker='o', title='P')
#df2.plot(x='x', y='y', kind='line', marker='o', ax=ax)
#dataf.plot(x='x', y='y', kind='line', marker='o', title='')

#plt.show()

print()
print(ax[100])
print(avg_Ax)
print(ax[100]-avg_Ax)
print(ax_calibrated[100])

# the direction of acceleration vector in the IMU frame of reference 
accel_IMUF = np.array([ax_calibrated, ay_calibrated, az_calibrated]).T
accel_IMUF_magnitude = np.linalg.norm(accel_IMUF, axis=1)
accel_direction_IMUF =  accel_IMUF / accel_IMUF_magnitude[:, np.newaxis]

# Check
#print(accel_direction_IMUF[5])
# The direction of angular rate vector in the IMU frame of reference
gyro_IMUF = np.array([gx_calibrated, gy_calibrated, gz_calibrated]).T
gyro_IMUF_magnitude = np.linalg.norm(gyro_IMUF, axis=1)
gyro_direction_IMUF =  gyro_IMUF / gyro_IMUF_magnitude[:, np.newaxis]


print('#####################################################')
print(gyro_IMUF.shape)
print(accel_IMUF[5])
print(accel_IMUF[5])


attitude_matrices = []
for i in range(len(mx)):
    acceleration = accel_IMUF[i]
    magnetic_field = magnetometer_IMUF_cal[i]
    attitude_matrix = earth_frame_conversion(acceleration, magnetic_field)
    attitude_matrices.append(attitude_matrix)

# Convert list to array
attitude_matrices = np.array(attitude_matrices)

print("Attitude Matrices:")
print(attitude_matrices)











#earth's frame in the IMU frame of reference

#North_EARTHF = np.cross()

'''
# Define the axis and angle
axis = np.array([0, 0, 1])  # Define as a NumPy array
angle_deg = 90

# Create a quaternion from axis-angle representation
quaternion = Rotation.from_rotvec(axis * np.radians(angle_deg))

# Access quaternion properties
print("Quaternion:")
print(quaternion)

# Access rotation matrix
rotation_matrix = quaternion.as_matrix()
print("\nRotation matrix:")
print(rotation_matrix)

# Access Euler angles
euler_angles = quaternion.as_euler('xyz', degrees=True)
print("\nEuler angles (in degrees):")
print(euler_angles)

'''

'''
calibration_matrix = np.array([[1.0, 0.02, -0.03],
                                [0.02, 0.98, 0.05],
                                [-0.03, 0.05, 1.1]])

# Sample raw magnetometer data (replace this with your actual data):
raw_data = np.array([[100, 20, -30],
                    [110, 25, -28],
                    [95, 18, -35]])

# Calibrate magnetometer data
calibrated_data = calibrate_magnetometer(raw_data, calibration_matrix)

print("Calibrated Magnetometer Data:")
print(calibrated_data)





acceleration = np.array([0.1, 0.2, 9.8])  # Example accelerometer data
magnetic_field = np.array([20, 30, -15])  # Example magnetometer data
attitude = np.eye(3)  # Example attitude (identity matrix for simplicity)

# Convert to Earth frame
earth_acceleration, heading_angle = earth_frame_conversion(acceleration, magnetic_field, attitude)

print("Earth Acceleration:", earth_acceleration)
print("Heading Angle (in radians):", heading_angle)


'''
