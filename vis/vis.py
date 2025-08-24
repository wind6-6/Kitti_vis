import os
import numpy as np
import open3d as o3d
from vision_utils import draw_lidar_with_box_colors, draw_box3d_lidar, gen_3dbox

# 全局变量用于控制可视化流程
current_index = 0
max_index = 9  # 显示范围为0-9
vis = None


class Kitti:
    def __init__(self, root_path="D:\\1study", ind=0) -> None:
        self.root_path = root_path
        self.ind = ind 
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
        if not os.path.exists(label_path):
            print(f"[错误] 标签文件不存在: {label_path}")
            return anns
        
        try:
            with open(label_path, 'r') as lf:
                labels = [label.rstrip() for label in lf.readlines()]
            
            for i, label in enumerate(labels):
                ann = label.split()
                # 过滤非当前帧
                if ann[0] != str(self.ind):
                    continue
                # 检查字段数量是否足够
                if len(ann) < 17:
                    print(f"[警告] 行 {i} 字段不足，跳过（实际: {len(ann)}, 预期: ≥17）")
                    continue
                # 过滤DontCare类型
                class_name = ann[2]
                if class_name == "DontCare":
                    continue
                
                try:
                    track_id = int(ann[1])
                    nums = [float(x) for x in ann[3:17]]
                except ValueError as e:
                    print(f"[警告] 行 {i} 数值转换失败: {str(e)}")
                    continue
                
                if len(nums) < 14:
                    print(f"[警告] 行 {i} 数值字段不足，跳过（实际: {len(nums)}, 预期: 14）")
                    continue
                    
                ann_format = {
                    "class_name": class_name,
                    "track_id": track_id,
                    "truncation": nums[0],
                    "occlusion": nums[1],
                    "alpha": nums[2],
                    "box2d": np.array([nums[3], nums[4], nums[5], nums[6]]),
                    "box3d": {
                        "dim": np.array([nums[9], nums[8], nums[7]]),
                        "center": np.array([nums[10], nums[11], nums[12]]),
                        "rotation": nums[13]
                    }
                }
                anns.append(ann_format)
                print(f"[调试] 成功解析行 {i}，类别: {class_name}")
        
        except Exception as e:
            print(f"[错误] 标签文件解析失败: {str(e)}")
        
        print(f"[调试] 最终有效标注数量: {len(anns)}")
        return anns


class VisKitti:
    def __init__(self, root_path="D:\\1study", ind=0) -> None:
        self.kitti = Kitti(root_path=root_path, ind=ind)
        self.calib = self.kitti.get_calib()
        self.anns = self.kitti.get_anns()
        self.lidar = self.kitti.get_lidar()

    def get_3dbox_data(self):
        """获取3D框数据"""
        if not self.anns:
            print("[错误] 没有可显示的标注数据")
            return None
        
        try:
            bbox = [ann["box3d"] for ann in self.anns]
            print(f"[调试] 提取到 {len(bbox)} 个3D框信息")
            bbox3d = gen_3dbox(bbox3d=bbox)
            if not bbox3d:
                print("[警告] gen_3dbox返回空数据")
                return None
            return bbox3d
        except Exception as e:
            print(f"[错误] 生成3D框失败: {str(e)}")
            return None

def load_next(vis):
    """按键回调函数：按w加载下一张"""
    global current_index, max_index
    
    # 切换到下一个索引
    current_index += 1
    if current_index > max_index:
        print("[提示] 已到达最后一张")
        current_index = max_index
        return False
    
    # 清除当前可视化内容（关键修复）
    vis.clear_geometries()
    vis.update_geometry(None)  # 强制更新几何状态
    vis.poll_events()
    vis.update_renderer()
    
    # 加载新数据
    print(f"\n[提示] 加载第 {current_index} 张图")
    vis_kitti = VisKitti(ind=current_index)
    
    # 检查数据有效性
    if vis_kitti.lidar is None or len(vis_kitti.lidar) == 0:
        print("[错误] 点云数据为空，跳过该帧")
        return False
    if not vis_kitti.calib:
        print("[错误] 标定数据为空，跳过该帧")
        return False
    
    # 获取3D框
    bbox3d = vis_kitti.get_3dbox_data()
    if not bbox3d:
        print("[错误] 3D框数据无效，跳过该帧")
        return False
    
    # 添加新的可视化内容
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    
    vis, all_lidar_box3d = draw_box3d_lidar(bbox3d, vis_kitti.calib, vis)
    vis = draw_lidar_with_box_colors(vis_kitti.lidar, vis, all_lidar_box3d)
    
    # 更新可视化（关键修复）
    vis.update_geometry(None)
    vis.poll_events()
    vis.update_renderer()
    return False


def main():
    global vis, current_index
    
    # 初始化第一个可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=600)
    
    # 注册按键回调：按'w'键加载下一张
    vis.register_key_callback(ord("W"), load_next)
    vis.register_key_callback(ord("w"), load_next)  # 同时支持大写W和小写w
    
    # 加载初始帧
    print(f"[提示] 加载第 {current_index} 张图 (按w键加载下一张)")
    vis_kitti = VisKitti(ind=current_index)
    
    # 检查初始数据
    if vis_kitti.lidar is None or len(vis_kitti.lidar) == 0:
        print("[错误] 初始点云数据为空")
        return
    if not vis_kitti.calib:
        print("[错误] 初始标定数据为空")
        return
    
    # 获取初始3D框
    bbox3d = vis_kitti.get_3dbox_data()
    if not bbox3d:
        print("[错误] 初始3D框数据无效")
        return
    
    # 添加初始可视化内容
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    
    vis, all_lidar_box3d = draw_box3d_lidar(bbox3d, vis_kitti.calib, vis)
    vis = draw_lidar_with_box_colors(vis_kitti.lidar, vis, all_lidar_box3d)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
