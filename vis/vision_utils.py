import os
import cv2
import numpy as np
import open3d as o3d


def is_point_in_box(point, box_corners):
    """判断点是否在3D bounding box内"""
    # 计算3D框的边向量
    v1 = box_corners[1] - box_corners[0]
    v2 = box_corners[3] - box_corners[0]
    v3 = box_corners[4] - box_corners[0]
    
    # 计算点到原点的向量
    p = point - box_corners[0]
    
    # 计算点在三个边向量上的投影
    dot1 = np.dot(p, v1)
    dot2 = np.dot(p, v2)
    dot3 = np.dot(p, v3)
    
    # 计算边向量的点积
    v1_dot = np.dot(v1, v1)
    v2_dot = np.dot(v2, v2)
    v3_dot = np.dot(v3, v3)
    
    # 判断点是否在3D框内
    return (0 <= dot1 <= v1_dot) and (0 <= dot2 <= v2_dot) and (0 <= dot3 <= v3_dot)


def draw_lidar_with_box_colors(pc, vis, boxes3d=None):
    """绘制点云，3D框内的点用不同颜色显示，背景点淡化"""
    points = pc[:, :3]
    points_intensity = pc[:, 3] * 255  # intensity
    
    # 创建点云对象
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    
    # 初始化点云颜色（背景点使用低饱和度灰色，降低存在感）
    points_colors = np.zeros([points.shape[0], 3])
    for i in range(points_intensity.shape[0]):
        # 背景点使用接近白色的浅灰色，降低对比度
        gray_val = 0.9 + (points_intensity[i]/255) * 0.1  # 0.9~1.0之间
        points_colors[i, :] = [gray_val, gray_val, gray_val]
    
    # 如果有3D框，修改框内点的颜色（高饱和度颜色突出显示）
    if boxes3d is not None and len(boxes3d) > 0:
        # 定义不同框的颜色（高饱和度颜色，突出显示）
        # 将原黄色[1,1,0]替换为绿色[0,1,0]，其他颜色保持不变
        box_colors = [
            [1, 0, 0],    # 红色
            [0, 0, 1],    # 蓝色
            [0, 1, 0],    # 绿色（替换了原来的黄色）
            [0, 1, 1],    # 青色
            [1, 0, 1]     # 紫色
        ]
        
        for box_idx, box_corners in enumerate(boxes3d):
            # 为每个框循环检查所有点
            for i in range(points.shape[0]):
                if is_point_in_box(points[i], box_corners):
                    # 循环使用预定义的高饱和度颜色
                    color_idx = box_idx % len(box_colors)
                    points_colors[i, :] = box_colors[color_idx]
    
    pointcloud.colors = o3d.utility.Vector3dVector(points_colors)
    
    # 设置点云渲染参数
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # 纯白色背景，降低干扰
    opt.point_size = 2.0  # 点大小保持不变
    vis.add_geometry(pointcloud)
    return vis


def draw_gt_boxes3d(gt_boxes3d, vis):
    """绘制3D bounding boxes，使用更粗的线条和高对比度颜色"""
    num = len(gt_boxes3d)
    for n in range(num):
        points_3dbox = gt_boxes3d[n]
        lines_box = np.array([[0, 1], [1, 2], [2, 3],[0, 3], [4, 5], [5, 6], [6, 7], [4, 7], 
                        [0, 4], [1, 5], [2, 6], [3, 7]])  # 指明哪两个顶点之间相连
        
        # 为3D框设置高对比度颜色（深蓝色）
        colors = np.array([[0, 0.6, 1] for j in range(len(lines_box))])  # 亮蓝色，更醒目
        
        line_set = o3d.geometry.LineSet()  # 创建line对象
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.points = o3d.utility.Vector3dVector(points_3dbox)
        
        # 增加线宽（通过修改渲染选项中的线宽）
        vis.add_geometry(line_set)
    
    # 调整线宽，让3D框更突出
    opt = vis.get_render_option()
    opt.line_width = 5.0  # 增大线宽，默认通常是1.0
    return vis


def draw_box3d_lidar(bbox3d, calib, vis):
    # method 1
    lidar2cam = calib["lidar2cam"]
    lidar2cam = expand_matrix(lidar2cam)
    cam2rect_ = calib["rect2ref"]
    cam2rect = np.eye(4, 4)
    cam2rect[:3, :3] = cam2rect_
    lidar2rec = np.dot(lidar2cam, cam2rect)
    rec2lidar = np.linalg.inv(lidar2rec) #(AB)-1 = B-1@A-1
    
    all_lidar_box3d = []
    for box3d in bbox3d:
        if np.any(box3d[2, :] < 0.1):
            continue
        box3d = np.concatenate([box3d, np.ones((1, 8))], axis=0)
        lidar_box3d = np.dot(rec2lidar, box3d)[:3, :]
        lidar_box3d = np.transpose(lidar_box3d)
        all_lidar_box3d.append(lidar_box3d)
    vis = draw_gt_boxes3d(all_lidar_box3d, vis)
    return vis, all_lidar_box3d  # 返回转换后的3D框用于颜色标记


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=1.0):
    """ Filter lidar points, keep those in image FOV """
    lidar2cam = calib["lidar2cam"]
    lidar2cam = expand_matrix(lidar2cam)
    cam2rect_ = calib["rect2ref"]
    cam2rect = np.eye(4, 4)
    cam2rect[:3, :3] = cam2rect_
    lidar2rec = np.dot(cam2rect, lidar2cam)
    P = calib["rect2image"]
    P = expand_matrix(P)
    project_velo_to_image = np.dot(P, lidar2rec)
    
    pc_velo_T = pc_velo.T
    pc_velo_T = np.concatenate([pc_velo_T[:3,:], np.ones((1, pc_velo_T.shape[1]))], axis=0)
    
    project_3dbox = np.dot(project_velo_to_image, pc_velo_T)[:3, :]
    pz = project_3dbox[2, :]
    px = project_3dbox[0, :]/pz
    py = project_3dbox[1, :]/pz
    pts_2d = np.vstack((px, py)).T
    
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]

    return imgfov_pc_velo, pts_2d, fov_inds
    
    
def draw_2dbox(img, bbox, names=None, save=False):
    assert len(bbox)==len(names), "names not match bbox"
    color_map = {"Car":(0, 255, 0), "Pedestrian":(255, 0, 0), "Cyclist":(0, 0, 255)}
    for i, box in enumerate(bbox):
        name = names[i]
        if name not in color_map.keys():
            continue
        color = color_map[name]
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            2,
        )
        name_coord = (int(box[0]), int(max(box[1]-5, 0)))
        cv2.putText(img, name, name_coord, 
                    cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    cv2.imshow("image_with_2dbox", img)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("image_with_2dbox.jpg", img)


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def expand_matrix(matrix):
    new_matrix = np.eye(4, 4)
    new_matrix[:3, :] = matrix
    return new_matrix


def gen_3dbox(bbox3d):
    corners_3d_all = []
    for box in bbox3d:
        center = box["center"]
        l, w, h = box["dim"]
        angle = box["rotation"]
        R = roty(angle)
        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = np.dot(R, corners)
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        corners_3d_all.append(corners_3d)
    return corners_3d_all


def project_box3d(bbox3d, calib):
    P = calib["rect2image"]
    P = expand_matrix(P)
    project_xy = []
    project_z = []
    for box3d in bbox3d:
        if np.any(box3d[2, :] < 0.1):
            continue
        box3d = np.concatenate([box3d, np.zeros((1, 8))], axis=0)
        project_3dbox = np.dot(P, box3d)[:3, :]
        pz = project_3dbox[2, :]
        px = project_3dbox[0, :]/pz
        py = project_3dbox[1, :]/pz
        xy = np.stack([px, py], axis=1)
        project_xy.append(xy)
        project_z.append(pz)
    print(project_xy)    
    return project_xy, project_z


def draw_project(img, project_xy, save=False):
    color_map = {"Car":(0, 255, 0), "Pedestrian":(255, 0, 0), "Cyclist":(0, 0, 255)}
    for i, qs in enumerate(project_xy):
        color = (0, 255, 0) 
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, 1)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, 1)
            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, 1)

    cv2.imshow("image_with_projectbox", img)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("image_with_projectbox.jpg", img)