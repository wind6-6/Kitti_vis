
# 配置环境  
conda create -n kitti_vis python=3.8  
conda activate kitti_vis  

pip install opencv-python pillow scipy matplotlib
pip install open3d



# 代码功能
  显示带3D框的点云  

# 使用方法
cd vis
python vis.py

# 数据集路径
training
├── calib
|   |——0000.txt
|   |...
|——label_02
|   |——0000.txt
|   |...
|——velodyne
|   |——0000
|   |   |——000000.bin
|   |   |——000001.bin
|   |   ...

# 注
数据集格式同A40上cxtrack数据格式
可视化时，需要修改calib\0000.txt中内容，将R_rect,Tr_velo_cam,Tr_imu_velo修改成  R_rect: Tr_velo_cam: Tr_imu_velo:（ 原始数据没有':'）
作者只在windows上运行过此代码，未在linux运行过过
