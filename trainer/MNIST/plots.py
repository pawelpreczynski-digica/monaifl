import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys
import os

from pathlib import Path
home = str(Path.home())

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
    
    ax1.plot(xs, ys, color='#444444', label='Model Loss')
    ax2.plot(xs, zs, color='#2494CC', label='Model Accuracy')

    ax1.legend()
    ax1.set_title("Model Training Monitor", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=16)

    ax2.legend()
    ax2.set_ylabel("Accuracy(%)", fontsize=16)
    ax2.set_xlabel("No of Epochs", fontsize=16)

ani = animation.FuncAnimation(fig, animate, interval=1000)
#plt.tight_layout()
plt.show()
