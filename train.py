import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# train.py 开头添加
#import sys
#sys.path.append(r"D:\wenjian\deepose\ultralytics-yolo11-20251008\ultralytics-yolo11-main\ultralytics\nn\modules\DAttention.py")
# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。

# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!

# 在20250502更新中，修改保存权重的逻辑，训练结束(注意是正常训练结束后，手动停止的没有)后统一会保存4个模型，
# 分别是best.pt、last.pt、best_fp32.pt、last_fp32.pt，其中不带fp32后缀的是fp16格式保存的，
# 但由于有些模块对fp16非常敏感，会出现后续使用val.py的时候精度为0的情况，这种情况下可以用后缀带fp32去测试。

# 想找到哪些yaml是做轻量化的话可以用get_all_yaml_param_and_flops.py脚本，这个脚本里面有对应的教程视频。

# YOLO11配置文件路径：ultralytics/cfg/models/11
# YOLO12配置文件路径：ultralytics/cfg/models/12 预训练权重在这里下:https://github.com/sunsmarterjie/yolov12 Turbo版本
# YOLO13配置文件路径：ultralytics/cfg/models/13 预训练权重在这里下:https://github.com/iMoonLab/yolov13

if __name__ == '__main__':
    model = YOLO(r"D:\wenjian\deepose\ultralytics-yolo11-20251008\ultralytics-yolo11-main\ultralytics\cfg\models\11\yolo11.yaml") # YOLO11
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r"D:\wenjian\1data\data.yaml",
               task='detect',
               epochs=100,
               batch=8,         # ✔️ 建议改8，4070S 12G更稳，显存绝对够用，精度无损失
               imgsz=640,
               device=0,
               optimizer='SGD',
               patience=100,
               pretrained=True,
               project='runs/train',
               name='exp',
               cache=False,      # ✔️ 强制禁用缓存，彻底杜绝脏缓存问题，重中之重！
               workers=0,        # ✔️ Windows系统必加，关闭多线程，完美解决路径读取错乱！
               single_cls=False, # ✔️ 关闭单类检测，防止类别解析错误
               rect=False        # ✔️ 关闭矩形训练，新手必关，避免读取异常
               )
     