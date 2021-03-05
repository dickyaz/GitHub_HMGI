import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

data = pd.read_csv('jabodetabek_grav.txt')

X = data['X'].drop_duplicates()
Y = data['Y'].drop_duplicates()

G = data['G']
g = G.to_numpy()
g = g.reshape(len(Y), len(X))
# plt.imshow(g)
# plt.show()
#
# #matrix kernel operator average windows 3x3
# k = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
# avg = signal.convolve2d(g, k, boundary='symm', mode='same')
# plt.imshow(avg)
# plt.show()

win = np.ones(15*15)
win = win.reshape(15,15)/(15*15)
# print(win)

avg = signal.convolve2d(g, win, boundary='symm', mode='same')
plt.imshow(avg)
plt.show()