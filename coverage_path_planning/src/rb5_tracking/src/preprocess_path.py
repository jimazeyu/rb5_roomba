import os
import numpy as np
import pickle

data = np.genfromtxt(r"c:\Users\alber\Desktop\HW5\output\path.txt", delimiter=",")
# print(data)

mapsize = 1
grid = 5
offset = 0.3

data = data * (mapsize/(grid*2))
data = np.vstack((data, data[0,:]))
print(data.shape)

theta_list = []

data_list = []
i = 0
for dt in data:
    if i == 0:
        data_list.append(dt+np.array([offset, offset]))
        i += 1
    else:
        if (dt == data[i-1]).all():
            i += 1
        else:
            data_list.append(dt+np.array([offset, offset]))
            i += 1
data = np.array(data_list)
print(data.shape)

for i in range(1, data.shape[0]):
    vec = data[i] - data[i-1]
    theta_list.append(np.arctan2(vec[1], vec[0]))

theta_list = np.array([0] + theta_list)
theta_list = np.expand_dims(theta_list, axis=1)
data = np.hstack((data, theta_list))


with open(r"c:\Users\alber\Desktop\HW5\output\path.pkl", "wb") as file:
    pickle.dump(data, file, protocol=2)

with open(r"c:\Users\alber\Desktop\HW5\output\path.pkl", "rb") as file:
    data = pickle.load(file)

print(data)