import numpy as np
import cv2

# 색상 설정
color_bg = (5, 5, 5)
color_fg = (128, 128, 128)

# 다각형 그리기 함수
def draw_polygon(num_sides, width, height):
    # 중심점 설정 (정 가운데)
    center = (width // 2, height // 2)
    
    # 반지름 설정
    radius = min(width, height) // 4

    # 임의의 회전 각도 생성
    angle = np.random.randint(0, 360)
    
    # 꼭지점 좌표 계산
    points = []
    for i in range(num_sides):
        x = center[0] + int(radius * np.cos(np.radians(i * 360/num_sides + angle)))
        y = center[1] + int(radius * np.sin(np.radians(i * 360/num_sides + angle)))
        points.append([x, y])
    
    # 다각형 그리기
    img = np.full((height, width, 3), color_bg, dtype=np.uint8)
    cv2.fillPoly(img, np.array([points], dtype=np.int32), color_fg)
    
    return img

# 데이터 생성 함수
def generate_data(num_images_per_shape):
    images = []
    labels = []
    width, height = 32, 32
    
    shapes = [3, 5, 6, 7, 8]
    
    for shape in shapes:
        for i in range(num_images_per_shape):
            # 다각형 그리기
            img = draw_polygon(shape, width, height)
            
            # 라벨링
            label = shape - 3
            if label == 0:
                images.append(img)
                labels.append(label+1)
            else:
                images.append(img)
                labels.append(label)
    
    return images, labels

# 데이터 생성
num_images_per_shape = 12000
train_images, train_labels = generate_data(num_images_per_shape)

# 이미지와 라벨 섞기
shuffle_idx = np.random.permutation(len(train_images))
train_images = np.array(train_images)[shuffle_idx]
train_labels = np.array(train_labels)[shuffle_idx]

# 데이터 저장
for i in range(len(train_images)):
    filename = f"../data/training/{i}.jpg"
    cv2.imwrite(filename, train_images[i])
    np.savetxt(f"../data/training/{i}.txt", [train_labels[i]], fmt="%d")
