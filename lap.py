import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_laplacian_pyramid(img, levels=4):
    gaussian_pyramid = [img.copy()]
    for i in range(levels - 1):
        img = cv2.pyrDown(img)
        gaussian_pyramid.append(img)
    
    laplacian_pyramid = []
    for i in range(levels - 1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid

def visualize_laplacian_pyramid_color(lap_pyramid, save_dir="./lap_final_output"):
    os.makedirs(save_dir, exist_ok=True)

    for i, lap in enumerate(lap_pyramid):
        lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        lap_color = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)

        # ===================== 强制把底色变成纯黑 =====================
        mask = lap_norm < 15  # 暗区域 = 背景
        lap_color[mask] = [0, 0, 0]  # 直接涂黑

        # 锐化强化线条
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        lap_color = cv2.filter2D(lap_color, -1, kernel)

        cv2.imwrite(os.path.join(save_dir, f"L{i}_final.png"), lap_color)
        print(f"✅ L{i} 已保存")

    # 拼接总图
    vis_list = []
    for lap in lap_pyramid:
        lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        lap_color = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
        
        mask = lap_norm < 15
        lap_color[mask] = [0, 0, 0]

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        lap_color = cv2.filter2D(lap_color, -1, kernel)
        vis_list.append(lap_color)

    max_w = max([im.shape[1] for im in vis_list])
    total_h = sum([im.shape[0] for im in vis_list])
    total_img = np.zeros((total_h, max_w, 3), dtype=np.uint8)

    y = 0
    for im in vis_list:
        h, w = im.shape[:2]
        total_img[y:y+h, 0:w] = im
        y += h

    cv2.imwrite(os.path.join(save_dir, "laplacian_total_final.png"), total_img)

    plt.figure(figsize=(6,10))
    plt.imshow(cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    img_path = "tupian1.jpg"
    img = cv2.imread(img_path, 0)
    lap_pyramid = generate_laplacian_pyramid(img, levels=4)
    visualize_laplacian_pyramid_color(lap_pyramid)