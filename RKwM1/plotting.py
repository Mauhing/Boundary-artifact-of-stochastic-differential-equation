import numpy as np
from matplotlib import pyplot as plt

hist_RS = np.load("hist_RSl.npy")
#hist_RS = hist_RS/np.mean(hist_RS)

hist_EM = np.load("hist_EMl.npy")
#hist_EM = hist_EM/np.mean(hist_EM)

midpoints = np.load("midl.npy")

plt.plot(hist_RS, midpoints)
plt.plot(hist_EM, midpoints)
plt.gca().invert_yaxis()
plt.show()
