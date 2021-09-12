import matplotlib.pyplot as plt
from statistics import mean
import csv
import numpy as np

with open('./trajectory_x.csv') as f:
    reader = csv.reader(f)
    centers_x = [int(row[0]) for row in reader]

with open('./trajectory_y.csv') as f:
    reader = csv.reader(f)
    centers_y = [int(row[0]) for row in reader]

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

query = np.array(centers_x)
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
# template2 = centers_x[248:1282]
# template3 = centers_x[1537:1678]
# template = centers_x[248:1282][122:153]

X = query_vel
Y = template1_vel
plt.axvspan(122, 153, color = (0.05, 1., 0.01, .3))
plt.axvspan(248, 1282, color = (0.05, 1., 0.01, .3))
plt.axvspan(1375, 1678, color = (0.05, 1., 0.01, .3))


# plt.axvspan(270, 300, color = (0.05, 0.01, 1., .3))
# plt.axvspan(467, 520, color = (0.05, 0.01, 1., .3))
# plt.axvspan(722, 729, color = (0.05, 0.01, 1., .3))
# plt.axvspan(1126, 1165, color = (0.05, 0.01, 1., .3))

Y_ = [template1_vel, template2_vel, template3_vel, template4_vel]
C_ = ["C1", "C2", "C3", "C5"]
E_ = [250, 2000, 2500, 3800]
pathes =[]

plt.plot(X)
for Y, C, E in zip(Y_, C_, E_):
    # plt.plot(Y)
    for path, cost in spring(X, Y, E):
    #     # for line in path:
    #     #     plt.plot(line, [X[line[0]], Y[line[1]]], linewidth=0.2, c="gray")
        plt.plot(path[:,0], X[path[:,0]], C="C1")
        pathes.extend(path[:,0])
plt.show()
print(pathes)

# data = np.zeros(len(query))
# for i in range(len(query)):
#     if i in pathes:
#         data[i] = 1
#     else:
#         data[i] = 0   

# plt.scatter([i for i in range(len(data))],data, alpha=[i for i in data], s=2)                                                                                                                                                                              
# plt.show()

# mean_ = mean(template)
# template = [cx if cx != 0 else mean_ for cx in template]
# template = np.array(template)
# template_vel = np.diff(template)
# # tmp = template
# # while len(query)>len(tmp):
# #     template.extend(tmp)
# # template.extend(tmp)
# # template = template[:len(query)]

# from dtw import *
# from scipy.signal import savgol_filter
# alignment = dtw(query_vel, template_vel, step_pattern=asymmetric,keep_internals=True,open_end=True,open_begin=True)
# query_vel = savgol_filter(query_vel, 7, 3)
# plt.plot(query)
# # plt.plot(query_vel)
# plt.plot(query_vel)
# plt.axvspan(122, 153, color = (0.05, 1., 0.01, .3))
# plt.axvspan(248, 1282, color = (0.05, 1., 0.01, .3))
# plt.axvspan(1537, 1678, color = (0.05, 1., 0.01, .3))
# plt.show()


# alignment.plot()
# print(alignment.index1)
# for i, dist in enumerate(alignment.index2):
# plt.plot(alignment.index1, alignment.index2) 
# plt.axvspan(122, 153, color = (0.05, 1., 0.01, .3))
# plt.axvspan(248, 1282, color = (0.05, 1., 0.01, .3))
# plt.axvspan(1537, 1678, color = (0.05, 1., 0.01, .3))
# plt.show()

# alignmentOBE = dtw(query, template,
#                         keep_internals=True,
#                         step_pattern=asymmetric,
#                         open_end=True,open_begin=True)
# alignmentOBE.plot(type="twoway",offset=1)
# ## Display the warping curve, i.e. the alignment curve
# alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
# dtw(query, template, keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))\
#     .plot(type="alignment")

# # print(rabinerJuangStepPattern(6,"c"))
# # rabinerJuangStepPattern(6,"c").plot()

# print(alignment)

'''
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
A = centers_x
B = centers_x[248:1282]
tmp = B
while len(A)>len(B):
    B.extend(tmp)
B.extend(tmp)
B = B[:len(A)]
distance, path = fastdtw(A, B, dist=euclidean)
print(distance)
plt.plot(A)
plt.plot(B)
for i, j in path:
   plt.plot([i, j], [A[i], B[j]],color='gray', alpha=0.1, linestyle='dotted')
plt.legend(["query", "template"], fontsize=10, loc=2)
plt.show()
'''
# plt.axvspan(122, 153, color = (0.05, 1., 0.01, .3))
# plt.axvspan(248, 1282, color = (0.05, 1., 0.01, .3))
# plt.axvspan(1537, 1678, color = (0.05, 1., 0.01, .3))

# plt.show()