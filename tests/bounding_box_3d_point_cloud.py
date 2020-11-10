import numpy as np
import open3d as o3d

point_cloud_path = '../output.ply'

print("Testing IO for meshes ...")
mesh = o3d.io.read_point_cloud(point_cloud_path)
print(mesh)