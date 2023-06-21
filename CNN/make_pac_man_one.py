import os
import numpy as np
from PIL import Image, ImageDraw

def create_pacman_image(angle, arc_length=300, size=70):
    radius = size // 2
    img = Image.new('RGB', (size, size), (128, 128, 128))  # Background color set to gray here
    draw = ImageDraw.Draw(img)
    
    arc_start_angle = angle
    arc_end_angle = angle + arc_length
    draw.pieslice((0, 0, size, size), arc_start_angle, arc_end_angle, fill=(0, 0, 0))
    
    return img

def create_illusory_contour_dataset(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    arc_lengths = [310, 270, 310, 310]  # Keep the arc lengths constant

    canvas_size = 200
    pac_man_size = 70
    offset = (canvas_size - pac_man_size) // 8
    
    img = Image.new('RGB', (canvas_size, canvas_size), (128, 128, 128))  # Background color set to gray here
    
    pac_man_angles = [90,360,180,230]
    for angle_index, angle in enumerate(pac_man_angles):
        arc_length = arc_lengths[angle_index]
        pac_man = create_pacman_image(angle, arc_length, pac_man_size)
        if angle_index < 2:
            x = -pac_man_size // 16 + 15
        else:
            x = canvas_size - pac_man_size + pac_man_size // 16 - 15
        if angle_index % 2 == 0:
            y = offset
        else:
            y = canvas_size - pac_man_size - 15
            
        img.paste(pac_man, (x, y))
    
    filename = f'illusory_contour_{0}.png'
    img.save(os.path.join(save_dir, filename))

if __name__ == '__main__':
    save_dir = '../data/pac-man'
    create_illusory_contour_dataset(save_dir)
