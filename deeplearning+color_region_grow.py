import open3d as o3d
import numpy as np
from helper_ply import read_ply

import open3d as o3d
import numpy as np
from sklearn import cluster
from copy import deepcopy
import open3d

def open3d_reconstruction(pcd):
    pcd.paint_uniform_color(color=[0, 0, 0])
    # 1. Alpha shapes轮廓提取
    pcd_rc1 = deepcopy(pcd)
    pcd_rc1.translate([-100, 0, 50])
    mesh_rc1 = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_rc1, alpha=6)
    mesh_rc1 = open3d.geometry.TriangleMesh(mesh_rc1)
    mesh_rc1.paint_uniform_color([1, 0, 0])     # 红色
    # 2. Ball pivoting滚球算法
    pcd_rc2 = deepcopy(pcd)
    pcd_rc2.translate([-50, 0, 50])
    pcd_rc2.estimate_normals(   # 法向量计算
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    radii = [2, 5, 8]
    mesh_rc2 = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd=pcd_rc2,
        radii=open3d.utility.DoubleVector(radii)
    )
    mesh_rc2 = open3d.geometry.TriangleMesh(mesh_rc2)
    mesh_rc2.paint_uniform_color([0, 1, 0])  # 绿色
    # 3. Poisson泊松曲面重建
    pcd_rc3 = deepcopy(pcd)
    pcd_rc3.translate([0, 0, 50])
    pcd_rc3.estimate_normals(   # 法向量计算
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    mesh_rc3, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_rc3, depth=12)
    mesh_rc3 = open3d.geometry.TriangleMesh(mesh_rc3)
    # 设置阈值去除低密度重建结果
    threshold_value = np.quantile(densities, 0.35)      # 寻找低密度列表中35%的分位数,返回的是数值而不是索引
    vertices_to_remove = densities < threshold_value    # 低于分位数的值设置为False, 依次对其进行消除
    mesh_rc3.remove_vertices_by_mask(vertices_to_remove)
    mesh_rc3.paint_uniform_color([0, 0, 1])
    # 4. Voxel体素重建
    pcd_rc4 = deepcopy(pcd)
    pcd_rc4.translate([50, 0, 50])
    pcd_rc4.paint_uniform_color(color=[0, 1, 1])
    mesh_rc4 = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd_rc4, voxel_size=2)
    # 点云可视化
    pcd.translate([-25, 0, 0])
    open3d.visualization.draw_geometries([pcd, mesh_rc1, mesh_rc2, mesh_rc3, mesh_rc4],
                                         window_name="rebuild",
                                         width=800,
                                         height=600)

def open3d_segment(pcd):
    pcd.paint_uniform_color(color=[0.5, 0.5, 0.5])
    plane_model, inliers = pcd.segment_plane(distance_threshold=1, ransac_n=10, num_iterations=1000)
    [A, B, C, D] = plane_model
    print(f"Plane equation: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0")
    colors = np.array(pcd.colors)
    colors[inliers] = [0, 0, 1]  # 平面内的点设置为蓝色
    pcd.colors = open3d.utility.Vector3dVector(colors)
    # 点云可视化
    open3d.visualization.draw_geometries([pcd],
                                         window_name="segment",
                                         width=800,
                                         height=600)


def open3d_sklearn_cluster(pcd):
    pcd.paint_uniform_color(color=[0, 0, 0])
    # 点云聚类
    points = np.array(pcd.points)
    dbscan = cluster.DBSCAN(eps=4, min_samples=20)      # 使用DBSCAN算法进行聚类
    dbscan.fit(points)
    labels = dbscan.labels_
    # 显示颜色设置
    colors = np.random.randint(0, 255, size=(max(labels) + 1, 3)) / 255    # 需要设置为n+1类，否则会数据越界造成报错
    colors = colors[labels]    # 很巧妙，为每个label直接分配一个颜色
    colors[labels < 0] = 0     # 噪点直接设置为0，用黑色显示
    pcd_cluster = deepcopy(pcd)
    pcd_cluster.translate([50, 0, 0])
    pcd_cluster.colors = open3d.utility.Vector3dVector(colors)
    # 点云可视化
    open3d.visualization.draw_geometries([pcd, pcd_cluster],
                                         window_name="sklearn cluster",
                                         width=800,
                                         height=600)

def region_grow_by_color(pcd):
    # 定义聚类生长算法参数
    criteria = o3d.geometry.RGBColorCriteria(min_bound=(0, 0, 0), max_bound=(255, 255, 255))
    cluster_tol = 0.05
    color_tol = 15
    min_cluster_size = 50
    max_cluster_size = 10000

    # 执行聚类生长算法
    labels = np.array(pcd.cluster_dbscan(eps=cluster_tol, min_points=min_cluster_size, rgb_criteria=criteria))

    # 可视化聚类结果
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(len(pcd.points), 3)))
    for i in range(len(pcd.points)):
        if labels[i] == -1:
            pcd.colors[i] = [0, 0, 0]
        else:
            pcd.colors[i] = o3d.utility.Vector3d(np.random.uniform(0, 1, size=3))
    o3d.visualization.draw_geometries([pcd])



if __name__=="__main__":
    filename="data/Area_1_conferenceRoom_1.ply"
    data=read_ply(filename)
    
    xyz=np.vstack((data['x'],data['y'],data['z'])).T
    rgb=np.vstack((data['red'],data['green'],data['blue'])).T
    
    labels=data['class']
    
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(xyz)
    pcd.colors=o3d.utility.Vector3dVector(rgb)
    
    #region_grow_by_color(pcd)
    open3d_sklearn_cluster(pcd)
    