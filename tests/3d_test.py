import numpy as np
from open3d import *    

def main():
	cloud = io.read_point_cloud("output_good.ply") # Read the point cloud
	downpcd = cloud.voxel_down_sample(voxel_size=0.05)
	visualization.draw_geometries([downpcd]) # Visualize the point cloud     

if __name__ == "__main__":
	main()