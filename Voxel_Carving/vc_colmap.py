import numpy as np
import open3d as o3d
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
from Voxel_Carving.read_write_model import read_model
from Voxel_Carving.vc_utils import project_points, pick_color_map, get_color


def Voxel_carve_colmap(dir, voxel_grid, mask, input_model, input_format):

    for index, voxel in enumerate(voxel_grid.get_voxels()):
        cen_pts = o3d.geometry.VoxelGrid.get_voxel_center_coordinate(voxel_grid, voxel.grid_index)
    cameras, images, _ = read_model(path=input_model, ext=input_format)
    Pixel_img=[]
    Pixel_mask=[]
    Cam = []
    img_r = []

    key= list(images.keys())
    for i in range(len(images)):

        img_path = images[key[i]].name
        Pixel_img.append(Image.open(os.path.join(dir, img_path)).load())
        img2 = Image.open(os.path.join(dir, img_path))
        
        img_mask = Image.open(os.path.join(mask, img_path)).convert('1') #lego
        img_mask = img_mask.resize((int(img2.size[0]), int(img2.size[1])))

        img_mask = np.asarray(img_mask, dtype='float32')
        Pixel_mask.append(Image.fromarray(np.asarray(img_mask)).load())
        h, w = img_mask.shape

        c = images[key[i]].camera_id
        K_matrix = np.array([cameras[c].params[0] , 0 , cameras[c].params[2], 0, cameras[c].params[1], cameras[c].params[3], 0, 0 ,1 ]).reshape(3,3)

        R_quart  = [images[key[i]].qvec[1], images[key[i]].qvec[2], images[key[i]].qvec[3], images[key[i]].qvec[0]]
        R_matrix = R.from_quat(R_quart).as_matrix().reshape(3,3)
        T_matrix = images[key[i]].tvec.reshape(3,1)
        P = np.hstack((R_matrix, T_matrix))
        P = np.dot(K_matrix, P)
        Cam.append(P)
        # print(P)
        img_r.append([w,h])


        # file_path = 'data/scan4/cam_col/' + img_path[:-4] + "_cam.txt"
        # save_camera_txt(R_matrix, T_matrix, K_matrix, file_path)
          
        extrinsic = np.vstack((np.hstack((R_matrix, T_matrix)), np.array([[0.0, 0.0, 0.0, 1.0]], dtype='float32')))
        camera_params = o3d.camera.PinholeCameraParameters()	
        camera_params.extrinsic = extrinsic
        camera_params.intrinsic.set_intrinsics(img_mask.shape[1], img_mask.shape[0], K_matrix[0,0],K_matrix[1,1], K_matrix[0,2], K_matrix[1,2])
        # print(camera_params.intrinsic.intrinsic_matrix)
        img_mask = o3d.geometry.Image(img_mask)
        voxel_grid.carve_silhouette(img_mask, camera_params)

    return {"carved_3d" : voxel_grid, "Image_info": {"Pixel_img": Pixel_img, "Cameras": Cam, "Pixel_mask": Pixel_mask,"Img_Res": img_r} }


def color_mapping(vc_dict):

    voxel_grid = vc_dict["carved_3d"]
    Image_info = vc_dict["Image_info"]
    
    vox_mesh=o3d.geometry.TriangleMesh()
    carved_grid = voxel_grid.get_voxels()

    v = 0
    for vox in carved_grid:
        cube = o3d.geometry.TriangleMesh.create_box(width=2, height=2,depth=2)

        cen_pts =  o3d.geometry.VoxelGrid.get_voxel_center_coordinate(voxel_grid, vox.grid_index)

        pts_2d  =  project_points(Image_info["Cameras"], cen_pts)
        # print(vox.grid_index, cen_pts, pts_2d)
        color  =   get_color(pts_2d, Image_info["Pixel_img"], Image_info["Pixel_mask"], Image_info["Img_Res"])
        color_select  = pick_color_map(color)
        cube.paint_uniform_color(color_select)
        cube.translate(vox.grid_index, relative=False)
        vox_mesh+=cube
        # v = v+1
        # print("Processed Voxels : ", v)
        v += 1
        
    return vox_mesh




