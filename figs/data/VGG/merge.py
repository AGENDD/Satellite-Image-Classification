import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fix = None
cyclic = None
oneCycle = None
reduceLROnPlateau = None

fix = pd.read_csv('figs/data/VGG/pvgg_VGG--128_200_0.0002.csv')
cyclic = pd.read_csv('figs/data/VGG/pvgg_VGG--128_200_CyclicLR.csv')
oneCycle = pd.read_csv('figs/data/VGG/pvgg_VGG--128_200_OneCycleLR.csv')
reduceLROnPlateau = pd.read_csv('figs/data/VGG/pvgg_VGG--128_200_ReduceLROnPlateau.csv')


x = fix[fix.columns[0]].values

fix = fix[fix.columns[1]].values
cyclic = cyclic[cyclic.columns[1]].values
oneCycle = oneCycle[oneCycle.columns[1]].values
reduceLROnPlateau = reduceLROnPlateau[reduceLROnPlateau.columns[1]].values

fix_y1 = np.max(fix)
fix_x1 = x[np.argmax(fix)]
cyclic_y2 = np.max(cyclic)
cyclic_x2 = x[np.argmax(cyclic)]
oneCycle_y2 = np.max(oneCycle)
oneCycle_x2 = x[np.argmax(oneCycle)]
reduceLROnPlateau_y2 = np.max(reduceLROnPlateau)
reduceLROnPlateau_x2 = x[np.argmax(reduceLROnPlateau)]




plt.figure()

LINEWIDTH = 0.8

plt.plot(x,fix,label='fix',color = 'red',linewidth = LINEWIDTH)
plt.plot(x,cyclic,label='cyclic',color = 'blue',linewidth = LINEWIDTH)
plt.plot(x,oneCycle,label='oneCycle',color = 'orange',linewidth = LINEWIDTH)
plt.plot(x,reduceLROnPlateau,label='reduceLROnPlateau',color = 'purple',linewidth = LINEWIDTH)

plt.axhline(fix_y1, linestyle='--', color='red',linewidth = LINEWIDTH)
plt.axvline(fix_x1, linestyle='--', color='red',linewidth = LINEWIDTH)
plt.axhline(cyclic_y2, linestyle='--', color='blue',linewidth = LINEWIDTH)
plt.axvline(cyclic_x2, linestyle='--', color='blue',linewidth = LINEWIDTH)
plt.axhline(oneCycle_y2, linestyle='--', color='orange',linewidth = LINEWIDTH)
plt.axvline(oneCycle_x2, linestyle='--', color='orange',linewidth = LINEWIDTH)
plt.axhline(reduceLROnPlateau_y2, linestyle='--', color='purple',linewidth = LINEWIDTH)
plt.axvline(reduceLROnPlateau_x2, linestyle='--', color='purple',linewidth = LINEWIDTH)

# plt.text(fix_x1, fix_y1, f'({fix_x1}, {fix_y1:.4f})', ha='right')
# plt.text(cyclic_x2, cyclic_y2, f'({cyclic_x2}, {cyclic_y2:.4f})', ha='left')

xticks = list(plt.gca().get_xticks())
yticks = list(plt.gca().get_yticks())

# yticks.remove(0.9500000000000001)

# print(yticks)
xticks.append(fix_x1)
yticks.append(fix_y1)
xticks.append(cyclic_x2)
yticks.append(cyclic_y2)
xticks.append(oneCycle_x2)
yticks.append(oneCycle_y2)
xticks.append(reduceLROnPlateau_x2)
yticks.append(reduceLROnPlateau_y2)

plt.gca().set_xticks(xticks)
plt.gca().set_yticks(yticks)
# plt.yticks(rotation=45)
plt.tick_params(axis='y', labelsize=7)

plt.ylim(0.6, 1.0)  # 设置y轴的范围
plt.xlim(0,200)
plt.legend()
plt.ioff()
plt.show()