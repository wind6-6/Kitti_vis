import os
import numpy as np
import open3d as o3d
from vision_utils import draw_lidar_with_box_colors, draw_box3d_lidar, gen_3dbox


class Kitti:
    def __init__(self, root_path="./", ind=0) -> None:
        self.root_path = root_path
        self.name = f"{ind:06d}"
        print(f"[调试] 初始化Kitti类，文件名: {self.name}")

    def get_lidar(self):
        lidar_dir = os.path.join(self.root_path, "training", "velodyne", f"{int(self.name)//1000:04d}")
        lidar_path = os.path.join(lidar_dir, self.name + ".bin")
        print(f"[调试] 点云路径: {lidar_path}")
        if not os.path.exists(lidar_path):
            print(f"[错误] 点云文件不存在: {lidar_path}")
            return None
        try:
            lidar = np.fromfile(lidar_path, dtype=np.float32)
            lidar_data = lidar.reshape((-1, 4))
            print(f"[调试] 点云加载成功，形状: {lidar_data.shape}")
            return lidar_data
        except Exception as e:
            print(f"[错误] 点云加载失败: {str(e)}")
            return None
    
    def get_calib(self):
        calib_dir = os.path.join(self.root_path, "training", "calib")
        calib_name = f"{int(self.name)//1000:04d}.txt"
        calib_path = os.path.join(calib_dir, calib_name)
        print(f"[调试] 标定文件路径: {calib_path}")
        if not os.path.exists(calib_path):
            print(f"[错误] 标定文件不存在: {calib_path}")
            return None
        try:
            calib = {}
            with open(calib_path, 'r') as cf:
                infos = [x.rstrip() for x in cf.readlines() if x.rstrip()]
            for info in infos:
                key, value = info.split(":", 1)
                calib[key] = np.array([float(x) for x in value.split()])
            formatted = self.format_calib(calib)
            print(f"[调试] 标定文件加载成功，包含键: {formatted.keys()}")
            return formatted
        except Exception as e:
            print(f"[错误] 标定文件解析失败: {str(e)}")
            return None

    def format_calib(self, calib):
        try:
            calib_format = {
                "rect2image": calib["P2"].reshape([3, 4]),
                "lidar2cam": calib["Tr_velo_cam"].reshape([3, 4]),
                "rect2ref": calib["R_rect"].reshape([3, 3])
            }
            return calib_format
        except KeyError as e:
            print(f"[错误] 标定文件缺少关键键: {str(e)}")
            return None

    def get_anns(self):
        anns = []
        label_dir = os.path.join(self.root_path, "training", "label_02")
        label_name = f"{int(self.name)//1000:04d}.txt"
        label_path = os.path.join(label_dir, label_name)
        print(f"[调试] 标签文件路径: {label_path}")
        if not os.path.exists(label_path):
            print(f"[错误] 标签文件不存在: {label_path}")
            return anns
        
        try:
            with open(label_path, 'r') as lf:
                labels = [label.rstrip() for label in lf.readlines()]
            print(f"[调试] 标签文件共 {len(labels)} 行")
            
            for i, label in enumerate(labels):
                ann = label.split()
                # 过滤非当前帧（只处理frame=0的标注）
                if ann[0] != "0":
                    continue
                # 检查字段数量是否足够（至少17个字段）
                if len(ann) < 17:
                    print(f"[警告] 行 {i} 字段不足，跳过（实际: {len(ann)}, 预期: ≥17）")
                    continue
                # 过滤DontCare类型
                class_name = ann[2]
                if class_name == "DontCare":
                    continue
                
                try:
                    track_id = int(ann[1])
                    # 解析数值字段（从索引3开始，共14个字段）
                    nums = [float(x) for x in ann[3:17]]  # 严格控制解析范围，避免越界
                except ValueError as e:
                    print(f"[警告] 行 {i} 数值转换失败: {str(e)}")
                    continue
                
                # 检查数值字段数量
                if len(nums) < 14:
                    print(f"[警告] 行 {i} 数值字段不足，跳过（实际: {len(nums)}, 预期: 14）")
                    continue
                    
                ann_format = {
                    "class_name": class_name,
                    "track_id": track_id,
                    "truncation": nums[0],  # ann[3]
                    "occlusion": nums[1],   # ann[4]
                    "alpha": nums[2],       # ann[5]
                    "box2d": np.array([nums[3], nums[4], nums[5], nums[6]]),  # ann[6-9]
                    "box3d": {
                        # 修正维度和中心坐标的索引对应关系
                        "dim": np.array([nums[9], nums[8], nums[7]]),  # l, w, h 对应 ann[12,11,10]
                        "center": np.array([nums[10], nums[11], nums[12]]),  # cx, cy, cz 对应 ann[13-15]
                        "rotation": nums[13]  # yaw 对应 ann[16]
                    }
                }
                anns.append(ann_format)
                print(f"[调试] 成功解析行 {i}，类别: {class_name}")
        
        except Exception as e:
            print(f"[错误] 标签文件解析失败: {str(e)}")
        
        print(f"[调试] 最终有效标注数量: {len(anns)}")
        return anns


class VisKitti:
    def __init__(self, root_path="./", ind=0) -> None:
        self.kitti = Kitti(root_path=root_path, ind=ind)
        self.calib = self.kitti.get_calib()
        self.anns = self.kitti.get_anns()

    def show_lidar_with_3dbox(self):  # 正确：与 __init__ 同级缩进
        # 检查标注是否存在
        if not self.anns:
            print("[错误] 没有可显示的标注数据")
            return
        
        # 提取3D框信息
        try:
            bbox = [ann["box3d"] for ann in self.anns]
            print(f"[调试] 提取到 {len(bbox)} 个3D框信息")
            bbox3d = gen_3dbox(bbox3d=bbox)
            print(f"[调试] 生成3D框顶点数据，共 {len(bbox3d)} 个框")
            if not bbox3d:
                print("[警告] gen_3dbox返回空数据")
                return
        except Exception as e:
            print(f"[错误] 生成3D框失败: {str(e)}")
            return
        
        # 检查点云数据
        lidar = self.kitti.get_lidar()
        if lidar is None or len(lidar) == 0:
            print("[错误] 点云数据为空")
            return
        
        # 检查标定数据
        if not self.calib:
            print("[错误] 标定数据为空")
            return
        
        # 创建可视化窗口
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=800, height=600)
            print("[调试] 可视化窗口创建成功")
            
            # 添加坐标轴
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
            vis.add_geometry(mesh_frame)
            
            # 添加3D框并获取转换后的框坐标
            vis, all_lidar_box3d = draw_box3d_lidar(bbox3d, self.calib, vis)
            print("[调试] 3D框已添加到可视化窗口")
            
            # 绘制点云（传入3D框以实现框内点着色）
            vis = draw_lidar_with_box_colors(lidar, vis, all_lidar_box3d)
            print("[调试] 点云已添加到可视化窗口")
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print(f"[错误] 可视化过程失败: {str(e)}")


if __name__ == "__main__":
    for i in range(0,153):
        vis = VisKitti(ind=i)
        vis.show_lidar_with_3dbox()
