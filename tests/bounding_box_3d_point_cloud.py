# import numpy as np
# import open3d as o3d

point_cloud_path = '../output.ply'

# print("Testing IO for meshes ...")
# mesh = o3d.io.read_point_cloud(point_cloud_path)
#print(mesh)

# import meshio

# mesh = meshio.read(point_cloud_path)
# meshio.save('output_mesh.ply')


from plyfile import PlyData, PlyElement

plydata = PlyData.read(point_cloud_path)

print(plydata.elements[0].name)
print(plydata.elements[0].data[0])
print(plydata.elements[0].data['x'])
print(plydata.elements[0].properties)