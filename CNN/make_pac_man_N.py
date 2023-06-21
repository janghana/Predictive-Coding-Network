import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

color_bg = (128, 128, 128)  # White background
color_fg = (0, 0, 0)  # Gray for vertices
color_poly = (128, 128, 128)  # Black for polygon

def draw_polygon(num_sides, width, height):
    center = (width // 2, height // 2)
    radius = min(width, height) // 4
    angle = np.random.randint(0, 360)
    points = []
    for i in range(num_sides):
        x = center[0] + int(radius * np.cos(np.radians(i * 360/num_sides + angle)))
        y = center[1] + int(radius * np.sin(np.radians(i * 360/num_sides + angle)))
        points.append([x, y])
    img = np.full((height, width, 3), color_bg, dtype=np.uint8)

    # Draw the vertices in gray
    for point in points:
        cv2.circle(img, tuple(point), 15//7, color_fg, -1)
    # Draw the polygon in black
    cv2.fillPoly(img, np.array([points], dtype=np.int32), color_poly)

    return img

def generate_data(num_images_per_shape):
    images = []
    labels = []
    width, height = 32, 32
    shapes = [3, 5, 6, 7, 8]
    shape_names = ["triangle", "pentagon", "hexagon", "heptagon", "octagon"]
    
    for shape, shape_name in zip(shapes, shape_names):
        for i in range(num_images_per_shape):
            img = draw_polygon(shape, width, height)
            label = shape - 3
            images.append(img)
            labels.append(label+1)
            filename = f"../data/pac-man/test/{shape_name}-pac_man.jpg"
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return images, labels

num_images_per_shape = 1
train_images, train_labels = generate_data(num_images_per_shape)
