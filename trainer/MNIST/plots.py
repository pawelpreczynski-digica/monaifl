import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 15))
#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#ax2 = fig.add_subplot(2,1,1)
def animate(i):
    graph_data = open('mnistlog.txt','r').read()
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
