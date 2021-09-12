import matplotlib.pyplot as plt
from statistics import mean
import csv
import numpy as np

with open('./trajectory_x.csv') as f:
    reader = csv.reader(f)
    centers_x = [int(row[0]) for row in reader]

with open('./trajectory_x_test5.csv') as f:
    reader = csv.reader(f)
    centers_x_test = [int(row[0]) for row in reader]

def dist(x, y):
    return (x - y)**2


def get_min(m0, m1, m2, i, j):
    if m0 < m1:
        if m0 < m2:
            return i - 1, j, m0
        else:
            return i - 1, j - 1, m2
    else:
        if m1 < m2:
            return i, j - 1, m1
        else:
            return i - 1, j - 1, m2

def partial_dtw(x, y):
    Tx = len(x)
    Ty = len(y)

    C = np.zeros((Tx, Ty))
    B = np.zeros((Tx, Ty, 2), int)

    C[0, 0] = dist(x[0], y[0])
    for i in range(Tx):
        C[i, 0] = dist(x[i], y[0])
        B[i, 0] = [0, 0]

    for j in range(1, Ty):
        C[0, j] = C[0, j - 1] + dist(x[0], y[j])
        B[0, j] = [0, j - 1]

    for i in range(1, Tx):
        for j in range(1, Ty):
            pi, pj, m = get_min(C[i - 1, j],
                                C[i, j - 1],
                                C[i - 1, j - 1],
                                i, j)
            C[i, j] = dist(x[i], y[j]) + m
            B[i, j] = [pi, pj]
    t_end = np.argmin(C[:,-1])
    cost = C[t_end, -1]
    
    path = [[t_end, Ty - 1]]
    i = t_end
    j = Ty - 1

    while (B[i, j][0] != 0 or B[i, j][1] != 0):
        path.append(B[i, j])
        i, j = B[i, j].astype(int)
        
    return np.array(path), cost

def spring(x, y, epsilon):
    Tx = len(x)
    Ty = len(y)

    C = np.zeros((Tx, Ty))
    B = np.zeros((Tx, Ty, 2), int)
    S = np.zeros((Tx, Ty), int)

    C[0, 0] = dist(x[0], y[0])

    for j in range(1, Ty):
        C[0, j] = C[0, j - 1] + dist(x[0], y[j])
        B[0, j] = [0, j - 1]
        S[0, j] = S[0, j - 1]
        
    for i in range(1, Tx):
        C[i, 0] = dist(x[i], y[0])
        B[i, 0] = [0, 0]
        S[i, 0] = i
        
        for j in range(1, Ty):
            pi, pj, m = get_min(C[i - 1, j],
                                C[i, j - 1],
                                C[i - 1, j - 1],
                                i, j)
            C[i, j] = dist(x[i], y[j]) + m
            B[i, j] = [pi, pj]
            S[i, j] = S[pi, pj]
            
        imin = np.argmin(C[:(i+1), -1])
        dmin = C[imin, -1]
        
        if dmin > epsilon:
            continue
            
        for j in range(1, Ty):
            if (C[i,j] < dmin) and (S[i, j] < imin):
                break
        else:
            path = [[imin, Ty - 1]]
            temp_i = imin
            temp_j = Ty - 1
            
            while (B[temp_i, temp_j][0] != 0 or B[temp_i, temp_j][1] != 0):
                path.append(B[temp_i, temp_j])
                temp_i, temp_j = B[temp_i, temp_j].astype(int)
                
            C[S <= imin] = 100000000
            yield np.array(path), dmin


from scipy.signal import savgol_filter

query = np.array(centers_x_test)
query_vel = np.diff(query)
query_vel = savgol_filter(query_vel, 11, 3)
query_acc = np.diff(query_vel)
template1 = np.array(centers_x[270:300])
template1_vel = np.diff(template1)
template1_vel = savgol_filter(template1_vel, 11, 3)
template2 = np.array(centers_x[470:529])
template2_vel = np.diff(template2)
template2_vel = savgol_filter(template2_vel, 17, 3)
template3 = np.array(centers_x[470:529])
template3_vel = np.diff(template3)
template3_vel = savgol_filter(template3_vel, 11, 3)
template4 = np.array(centers_x[1126:1165])
template4_vel = np.diff(template4)
template4_vel = savgol_filter(template4_vel, 11, 3)

# plt.axvspan(122, 153, color = (0.05, 1., 0.01, .3))
# plt.axvspan(248, 1282, color = (0.05, 1., 0.01, .3))
# plt.axvspan(1375, 1678, color = (0.05, 1., 0.01, .3))

plt.axvspan(206, 267, color = (0.05, 1., 0.01, .3))
plt.axvspan(318, 340, color = (0.05, 1., 0.01, .3))
plt.axvspan(390, 604, color = (0.05, 1., 0.01, .3))
plt.axvspan(604, 834, color = (0.05, 1., 0.01, .3))

# plt.axvspan(270, 300, color = (0.05, 0.01, 1., .3))
# plt.axvspan(467, 520, color = (0.05, 0.01, 1., .3))
# plt.axvspan(722, 729, color = (0.05, 0.01, 1., .3))
# plt.axvspan(1126, 1165, color = (0.05, 0.01, 1., .3))

X = query_vel
Y_ = [template1_vel, template2_vel, template3_vel, template4_vel]
C_ = ["C1", "C2", "C3", "C5"]
E_ = [180, 1800, 2300, 3800]
pathes =[]

plt.plot(X)
for Y, C, E in zip(Y_, C_, E_):
    # plt.plot(Y)
    for path, cost in spring(X, Y, E):
    #     # for line in path:
    #     #     plt.plot(line, [X[line[0]], Y[line[1]]], linewidth=0.2, c="gray")
        plt.plot(path[:,0], X[path[:,0]], C="C1")
        pathes.extend(path[:,0])
plt.title('Action Recognition by DTW')
plt.legend(['Input(Inproper handwash)', 'Detected'], loc='upper right')
plt.show()
# print(pathes)


