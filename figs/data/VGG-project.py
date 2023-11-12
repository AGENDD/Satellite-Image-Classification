import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

vgg = pd.read_csv('figs/data/VGG/pvgg_VGG--128_200_0.0002.csv')
p = pd.read_csv('figs/data/projects/p1_Model1--128_200_0.0002.csv')


x = p[p.columns[0]].values
p = p[p.columns[1]].values
vgg = vgg[vgg.columns[1]].values

p_maxy = max(p)
p_maxx = x[np.argmax(p)]
vgg_maxy = max(vgg)
vgg_maxx = x[np.argmax(vgg)]

plt.plot(x,p,label = 'small model')
plt.plot(x,vgg, label = 'large model')

LINEWIDTH = 0.8
plt.axhline(p_maxy, linestyle='--', color='blue',linewidth = LINEWIDTH)
plt.axvline(p_maxx, linestyle='--', color='blue',linewidth = LINEWIDTH)
plt.axhline(vgg_maxy, linestyle='--', color='orange',linewidth = LINEWIDTH)
plt.axvline(vgg_maxx, linestyle='--', color='orange',linewidth = LINEWIDTH)

xticks = list(plt.gca().get_xticks())
yticks = list(plt.gca().get_yticks())
print(xticks)


xticks.append(p_maxx)
yticks.append(p_maxy)
xticks.append(vgg_maxx)
yticks.append(vgg_maxy)
xticks.remove(-25.0)
plt.gca().set_xticks(xticks)
plt.gca().set_yticks(yticks)

plt.ylim(0.3, 1.0)  # 设置y轴的范围
plt.xlim(0,200)

plt.legend()
plt.savefig(f"figs/data/vgg_project_initial.png")
plt.show()
plt.ioff()
