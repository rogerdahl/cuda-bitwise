#!/usr/bin/env python

import random
import matplotlib.pyplot as plt

num_slots = 1 << 16
num_filled_slots = 0
slots = [False] * num_slots
num_tries = 0
plot_list = []

while num_filled_slots < num_slots:
    num_tries += 1
    rnd = random.randint(0, num_slots - 1)
    if not slots[rnd]:
        slots[rnd] = True
        num_filled_slots += 1
        # print "{} {} {}".format(n, c, completed_percent)
        # plot_list.append((completed_percent, num_tries))

    filled_slots_percent = float(num_filled_slots) / num_slots * 100
    plot_list.append(filled_slots_percent)

# Percent finished for given number of filled slots
# search_filled_slots_percent = float(64974) / num_slots * 100 # Bjorn
search_filled_slots_percent = float(63582) / num_slots * 100
for i, filled_slots_percent in enumerate(plot_list):
    if filled_slots_percent > search_filled_slots_percent:
        print 'Percent completed: {}'.format(float(i) / len(plot_list) * 100)
        break

#plt.plot(plot_list)
#plt.show()
