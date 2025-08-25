
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
│   ├── 0000.txt  
│   ├── ...  
├── label_02  
│   ├── 0000.txt  
│   ├── ...  
├── velodyne  
│   ├── 0000  
│   │   ├── 000000.bin  
│   │   ├── 000001.bin  
│   │   ├── ...  

# 注
可在windows上执行
