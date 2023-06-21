import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pacman(ax, center, radius, angle):
    arc = patches.Wedge(center, radius, angle * 180 / np.pi - 30, angle * 180 / np.pi + 30, color='black')
    ax.add_patch(arc)

def draw_v(ax, tip, height, angle):
    x1, y1 = tip[0] - height * np.cos(angle - np.pi / 6), tip[1] - height * np.sin(angle - np.pi / 6)
    x2, y2 = tip[0] - height * np.cos(angle + np.pi / 6), tip[1] - height * np.sin(angle + np.pi / 6)

    ax.plot([x1, tip[0], x2], [y1, tip[1], y2], 'k-', linewidth=2)

fig, ax = plt.subplots(figsize=(6, 6))

pac_radius = 1
v_height = 1

angles = np.arange(0, 2 * np.pi, 2 * np.pi / 3)
pac_centers = np.column_stack([2 * pac_radius * np.cos(angles), 2 * pac_radius * np.sin(angles)])

for i in range(3):
    draw_pacman(ax, pac_centers[i], pac_radius, angles[i] + np.pi / 2)
    draw_v(ax, pac_centers[i], v_height, angles[i])

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

plt.show()