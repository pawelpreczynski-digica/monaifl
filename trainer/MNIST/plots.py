import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys
import os
#from scipy.interpolate import spline
from pathlib import Path
home = str(Path.home())
import numpy as np
from scipy.interpolate import  interp1d, Rbf, InterpolatedUnivariateSpline

logpath = os.path.join(home, "monaifl", "trainer", "save","logs","client")
logName = 'mnistlog.txt'
logFile = os.path.join(logpath, logName)

style.use('seaborn')
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 15))

def animate(i):
    graph_data = open(logFile,'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    zs = []
    for line in lines:
        if len(line) > 1:
            x, y, z = line.split(',')
            xs.append(int(x)+1)
            ys.append(float(y))
            zs.append(float(z))
    ax1.clear()
    ax2.clear()

    xnew = np.linspace(0, 1, num=40, endpoint=True)
    
#    iusy = InterpolatedUnivariateSpline(xs, ys)
#    ynew = iusy(xnew)

#    iusz = InterpolatedUnivariateSpline(xs, zs)
#    znew = iusz(xnew)

#    rbfy = Rbf(xs, ys)
#    ynew = rbfy(xnew)

#    rbfz = Rbf(xs, zs)
#    znew = rbfz(xnew)

#    rbfy = interp1d(xs, ys, kind='cubic', bounds_error=False)
#    ynew = rbfy(xnew)

#    rbfz = interp1d(xs, zs, bounds_error=False)
#    znew = rbfz(xnew)

#   a_BSpline = interpolate.make_interp_spline(x, y) 
#    ynew = spline(xs, ys, xnew)
#    znew = spline(xs, zs, xnew)
    
    ax1.plot(xs, ys, color='#444444', label='Model Loss')
    ax2.plot(xs, zs, color='#2494CC', label='Model Accuracy')


#    ax1.plot(xnew, ynew, color='#444444', label='Model Loss')
#    ax2.plot(xnew, znew, color='#2494CC', label='Model Accuracy')

    ax1.legend()
    ax1.set_title("Model Training Monitor", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=16)

    ax2.legend()
    ax2.set_ylabel("Accuracy(%)", fontsize=16)
    ax2.set_xlabel("No of Epochs", fontsize=16)

ani = animation.FuncAnimation(fig, animate, interval=1000)
#plt.tight_layout()
plt.show()

# x_new = np.linspace(1, 4, 300)

# a_BSpline = interpolate.make_interp_spline(x, y)

# y_new = a_BSpline(x_new)


# plt.plot(x_new, y_new)

#from scipy.interpolate import spline

# 300 represents number of points to make between T.min and T.max
#xnew = np.linspace(T.min(), T.max(), 300)  

#power_smooth = spline(T, power, xnew)

#plt.plot(xnew,power_smooth)
#plt.show()

# import numpy as np
# from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
# import matplotlib.pyplot as plt


# # setup data
# x = np.linspace(0, 10, 9)
# y = np.sin(x)
# xi = np.linspace(0, 10, 101)

# # use fitpack2 method
# ius = InterpolatedUnivariateSpline(x, y)
# yi = ius(xi)

# plt.subplot(2, 1, 1)
# plt.plot(x, y, 'bo')
# plt.plot(xi, yi, 'g')
# plt.plot(xi, np.sin(xi), 'r')
# plt.title('Interpolation using univariate spline')

# # use RBF method
# rbf = Rbf(x, y)
# fi = rbf(xi)

# plt.subplot(2, 1, 2)
# plt.plot(x, y, 'bo')
# plt.plot(xi, fi, 'g')
# plt.plot(xi, np.sin(xi), 'r')
# plt.title('Interpolation using RBF - multiquadrics')
# plt.show()

