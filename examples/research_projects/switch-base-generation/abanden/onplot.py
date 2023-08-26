inid = [44, 54, 59, 61, 70, 72, 84, 86, 87, 90, 95, 101, 103, 108, 114, 116, 117, 120, 121, 122, 123, 124, 125, 126]

lo = []
for i in range(126+1):
    if i in inid:
        lo.append(1)
    else:
        lo.append(0)


import matplotlib.pyplot as plt
import os
plt.plot(lo)
plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/save.png")