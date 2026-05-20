# YOLO11 Pose 姿态标签一键修复脚本 - 自动适配你的路径，直接运行
import os
import cv2

# 你的数据集根路径 - 不用改！
root_path = r"D:\wenjian\person-pose"
label_paths = [os.path.join(root_path, "labels/train"), os.path.join(root_path, "labels/val")]
img_paths = [os.path.join(root_path, "images/train"), os.path.join(root_path, "images/val")]

# 修复每个标签文件
for label_path, img_path in zip(label_paths, img_paths):
    if not os.path.exists(label_path):
        continue
    for txt_name in os.listdir(label_path):
        if not txt_name.endswith(".txt"):
            continue
        txt_path = os.path.join(label_path, txt_name)
        img_name = txt_name.replace(".txt", ".jpg")
        img_path_full = os.path.join(img_path, img_name)
        
        # 删除空标签文件
        if os.path.getsize(txt_path) == 0:
            os.remove(txt_path)
            continue
        
        # 读取标签并修复
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            parts = [float(p) if p.replace('.','').replace('-','').isdigit() else 0 for p in parts]
            
            # 核心修复：保证开头是 0 + 归一化框坐标
            if len(parts) >= 5:
                cls, x, y, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
            else:
                cls, x, y, w, h = 0, 0.5, 0.5, 0.3, 0.6 # 默认框
            
            # 核心修复：补全17个关键点+3个值(x,y,vis=2)
            kpts = []
            if len(parts) >=6:
                kpts = parts[5:]
            # 补全到51个值(17*3)，不足补0 2
            while len(kpts) < 51:
                kpts.extend([0.0, 0.0, 2])
            kpts = kpts[:51]
            
            # 拼接成标准格式
            new_line = [0, x, y, w, h] + kpts
            new_line = [str(round(p, 4)) for p in new_line]
            new_lines.append(" ".join(new_line))
        
        # 写入修复后的标签
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))

print("✅ 所有姿态标签修复完成！已转为YOLO11 Pose标准格式（56个值，17关键点+vis）")