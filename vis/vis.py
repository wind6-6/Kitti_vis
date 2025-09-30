import os
import numpy as np
import open3d as o3d
from vision_utils import draw_lidar_with_box_colors, draw_box3d_lidar, gen_3dbox

# 全局变量用于控制可视化流程
current_index = 0
max_index = 153  # 显示范围为0-153
vis = None
# 复用的几何对象（避免重复创建）
current_pcd = None
current_boxes = []  # 存储当前帧的3D框几何对象
current_coord = None
# 保存相机参数的变量
camera_params = None  # 新增：用于保存相机参数


class Kitti:
    def __init__(self, root_path="./", ind=0) -> None:
        self.root_path = root_path
        self.ind = ind 
        self.name = f"{ind:06d}"

    def get_lidar(self):
        lidar_dir = os.path.join(self.root_path, "training", "velodyne", f"{int(self.name)//1000:04d}")
        lidar_path = os.path.join(lidar_dir, self.name + ".bin")
        if not os.path.exists(lidar_path):
            print(f"[错误] 点云文件不存在: {lidar_path}")
            return None
        try:
            lidar = np.fromfile(lidar_path, dtype=np.float32)
            lidar_data = lidar.reshape((-1, 4))  # 每行包含 (x, y, z, 反射强度r)
            
            # -------------------------- 关键修改1：输出10个点的坐标 --------------------------
            self.print_random_10_points(lidar_data)
            # ----------------------------------------------------------------------------------
            
            return lidar_data
        except Exception as e:
            print(f"[错误] 点云加载失败: {str(e)}")
            return None
    
    # -------------------------- 新增函数：打印10个随机点的坐标 --------------------------
    def print_random_10_points(self, lidar_data):
        """
        从点云数据中随机选取10个点，打印其(x, y, z)坐标（忽略反射强度r）
        :param lidar_data: 完整点云数据，形状为 (N, 4)，N为点的总数
        """
        total_points = len(lidar_data)
        if total_points == 0:
            print(f"[帧 {self.ind}] 点云数据为空，无坐标可输出")
            return
        
        # 若点云总数不足10个，直接输出所有点；否则随机采样10个点
        sample_size = min(10, total_points)
        # 生成随机索引（确保每次采样不重复）
        random_indices = np.random.choice(total_points, size=sample_size, replace=False)
        # 提取选中点的 (x, y, z) 坐标（第0-2列，第3列为反射强度）
        sampled_points = lidar_data[random_indices, :3]  # 形状为 (10, 3)
        
        # 格式化输出
        print(f"\n=== [帧 {self.ind}] 随机10个点的3D坐标 (x, y, z) ===")
        for i, (x, y, z) in enumerate(sampled_points, 1):
            # 保留6位小数（KITTI点云坐标精度通常在厘米级，6位小数足够）
            print(f"点{i:2d}: x={x:8.6f}, y={y:8.6f}, z={z:8.6f}")
        print(f"===============================================\n")
    # ----------------------------------------------------------------------------------
    
    def get_calib(self):
        calib_dir = os.path.join(self.root_path, "training", "calib")
        calib_name = f"{int(self.name)//1000:04d}.txt"
        calib_path = os.path.join(calib_dir, calib_name)
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
            return self.format_calib(calib)
        except Exception as e:
            print(f"[错误] 标定文件解析失败: {str(e)}")
            return None

    def format_calib(self, calib):
        try:
            return {
                "rect2image": calib["P2"].reshape([3, 4]),
                "lidar2cam": calib["Tr_velo_cam"].reshape([3, 4]),
                "rect2ref": calib["R_rect"].reshape([3, 3])
            }
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
            
            for label in labels:
                ann = label.split()
                if ann[0] != str(self.ind) or len(ann) < 17 or ann[2] == "DontCare":
                    continue
                try:
                    nums = [float(x) for x in ann[3:17]]
                    if len(nums) < 14:
                        continue
                    anns.append({
                        "class_name": ann[2],
                        "box3d": {
                            "dim": np.array([nums[9], nums[8], nums[7]]),
                            "center": np.array([nums[10], nums[11], nums[12]]),
                            "rotation": nums[13]
                        }
                    })
                except ValueError as e:
                    print(f"[警告] 标签解析失败: {str(e)}")
                    continue
        except Exception as e:
            print(f"[错误] 标签文件读取失败: {str(e)}")
            pass
        return anns


class VisKitti:
    def __init__(self, root_path="./", ind=0) -> None:
        self.kitti = Kitti(root_path=root_path, ind=ind)
        self.calib = self.kitti.get_calib()
        self.anns = self.kitti.get_anns()
        self.lidar = self.kitti.get_lidar()

    def get_3dbox_data(self):
        if not self.anns:
            print("[错误] 没有标注数据")
            return None
        try:
            bbox = [ann["box3d"] for ann in self.anns]
            return gen_3dbox(bbox3d=bbox)
        except Exception as e:
            print(f"[错误] 生成3D框失败: {str(e)}")
            return None


def draw_gt_boxes3d_modified(gt_boxes3d):
    """修改版3D框绘制函数，返回创建的LineSet对象列表"""
    line_sets = []
    num = len(gt_boxes3d)
    for n in range(num):
        points_3dbox = gt_boxes3d[n]
        lines_box = np.array([
            [0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])
        
        colors = np.array([[0, 0.6, 1] for _ in range(len(lines_box))])
        
        line_set = o3d.geometry.LineSet()
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.points = o3d.utility.Vector3dVector(points_3dbox)
        
        line_sets.append(line_set)
    return line_sets


def draw_box3d_lidar_modified(bbox3d, calib):
    """修改版3D框转换函数，返回转换后的框和LineSet对象"""
    lidar2cam = calib["lidar2cam"]
    lidar2cam = expand_matrix(lidar2cam)
    cam2rect_ = calib["rect2ref"]
    cam2rect = np.eye(4, 4)
    cam2rect[:3, :3] = cam2rect_
    lidar2rec = np.dot(lidar2cam, cam2rect)
    rec2lidar = np.linalg.inv(lidar2rec)
    
    all_lidar_box3d = []
    for box3d in bbox3d:
        if np.any(box3d[2, :] < 0.1):
            continue
        box3d = np.concatenate([box3d, np.ones((1, 8))], axis=0)
        lidar_box3d = np.dot(rec2lidar, box3d)[:3, :]
        lidar_box3d = np.transpose(lidar_box3d)
        all_lidar_box3d.append(lidar_box3d)
    
    # 生成LineSet对象
    line_sets = draw_gt_boxes3d_modified(all_lidar_box3d)
    return all_lidar_box3d, line_sets


def expand_matrix(matrix):
    """辅助函数：将3x4矩阵扩展为4x4"""
    new_matrix = np.eye(4, 4)
    new_matrix[:3, :] = matrix
    return new_matrix


def update_frame(vis, frame_id):
    """仅更新当前帧的几何数据，不重建可视化窗口，保持视角不变"""
    global current_pcd, current_boxes, current_coord, camera_params

    # 保存当前相机参数（如果已存在）
    if camera_params is not None:
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # 加载新帧数据
    vis_kitti = VisKitti(ind=frame_id)
    if vis_kitti.lidar is None or not vis_kitti.calib:
        return False
    bbox3d = vis_kitti.get_3dbox_data()
    if not bbox3d:
        return False

    # 1. 更新点云（复用对象，仅更新数据）
    point_xyz = vis_kitti.lidar[:, :3]  # 提取xyz坐标
# 过滤地面点
    point_xyz = filter_ground_points(point_xyz)  # 添加这行代码
    # 计算点云颜色
    def get_color(point_xyz):
        low = (0.6, 0.6, 0.6)
        high = (0.2, 0.2, 1.0)
        low_color = np.tile(np.array(low), (point_xyz.shape[0], 1))
        high_color = np.tile(np.array(high), (point_xyz.shape[0], 1))
        h = point_xyz[:, 1:2]  # 用y轴坐标映射颜色
        h = np.clip(h, np.mean(h)-1.5*np.std(h), np.mean(h)+1.5*np.std(h))
        alpha = (h - h.min())/(h.max()-h.min()+1e-6)
        return alpha * high_color + (1-alpha)*low_color

    point_color = get_color(point_xyz)
    
    if current_pcd is None:
        current_pcd = o3d.geometry.PointCloud()
        vis.add_geometry(current_pcd)
    current_pcd.points = o3d.utility.Vector3dVector(point_xyz)
    current_pcd.colors = o3d.utility.Vector3dVector(point_color)
    vis.update_geometry(current_pcd)

    # 2. 更新3D框（先移除旧框，再添加新框）
    for box in current_boxes:
        vis.remove_geometry(box)
    current_boxes.clear()
    
    # 绘制新框（使用修改后的函数获取LineSet对象）
    if bbox3d is not None:
        all_lidar_box3d, line_sets = draw_box3d_lidar_modified(bbox3d, vis_kitti.calib)
        for line_set in line_sets:
            vis.add_geometry(line_set)
            current_boxes.append(line_set)  # 记录新框引用

    # 3. 初始化坐标系（仅创建一次）
    if current_coord is None:
        current_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
        vis.add_geometry(current_coord)

    # 恢复相机参数（如果已保存）
    if camera_params is not None:
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

    # 更新渲染
    vis.update_geometry(current_coord)
    vis.poll_events()
    vis.update_renderer()
    return True


def switch_to_next_frame(vis):
    """切换到下一帧"""
    global current_index, max_index
    if current_index < max_index:
        current_index += 1
        update_frame(vis, current_index)
        print(f"切换到帧 {current_index}/{max_index}")
    else:
        print("已到达最后一帧")


def switch_to_prev_frame(vis):
    """切换到上一帧"""
    global current_index
    if current_index > 0:
        current_index -= 1
        update_frame(vis, current_index)
        print(f"切换到帧 {current_index}/{max_index}")
    else:
        print("已到达第一帧")


def exit_vis(vis):
    vis.destroy_window()

def filter_ground_points(points, threshold=-1.65):
    """过滤地面点，保留z轴坐标大于阈值的点（z轴为高度方向）"""
    return points[points[:, 2] > threshold] 

def main():
    global vis, current_index, camera_params

    # 初始化可视化窗口（仅创建一次）
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1600, height=800)
    
    # 设置渲染参数
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 2.0
    opt.line_width = 5.0

    # 注册按键回调（A键上一帧，D键下一帧，ESC退出）
    vis.register_key_callback(65, switch_to_prev_frame)   # A键（ASCII码65）
    vis.register_key_callback(68, switch_to_next_frame)   # D键（ASCII码68）
    vis.register_key_callback(27, exit_vis)               # ESC键（ASCII码27）

    # 加载初始帧
    update_frame(vis, current_index)
    # 初始帧加载完成后保存相机参数
    camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
