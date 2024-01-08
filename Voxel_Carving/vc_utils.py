import numpy as np

#Reading DTU camera files
def read_cam_file(filename , scale_x, scale_y):
    with open(filename) as k:
        lines = k.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3)) #dtu 7,10  #colmap=> 6:9
    # intrinsics = np.fromstring(' '.join(lines[0:3]), dtype=np.float32, sep=' ').reshape((3, 3)) 
    k.close()
    intrinsics[0,0] =  intrinsics[0,0] / scale_x
    intrinsics[0,2] =  intrinsics[0,2] / scale_x
    intrinsics[1,1] =  intrinsics[1,1] / scale_y
    intrinsics[1,2] =  intrinsics[1,2] / scale_y

    return intrinsics, extrinsics

#Reading Temple camera files
def read_cam_file_temple(filename1 , filename2):
    with open(filename1) as k:
        lines = k.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[0:4]), dtype=np.float32, sep=' ').reshape((4, 4))
    k.close()

    with open(filename2) as k:
        lines = k.readlines()
        lines = [line.rstrip() for line in lines]
    intrinsics = np.fromstring(' '.join(lines[0:3]), dtype=np.float32, sep=' ').reshape((3, 3)) 
    k.close()

    return intrinsics, extrinsics

def project_points(Cams ,point):
    Cams = np.vstack(np.array(Cams))
    point = np.hstack((point, 1))
    pt_2d = np.dot(Cams, point)
    pt_2d = pt_2d.reshape((int(pt_2d.shape[0]/3) , 3))
    pt_2d = pt_2d/ pt_2d[:,[-1]]
    return pt_2d


"""
Texturing 92초중 90초 사용하는 함수
Input : pt_2d - 그리드에 매핑된 이미지의 Color (DTU에선 48x3), IMG = 이미지 픽셀의 값, IMG_mask : 이미지 마스크(Pixel의 값), IMG_sh : 이미지 Resolution (DTU에선 640x512)
"""
def get_color(pt_2d, Img, Img_mask,img_sh):
        indx = list(range(len(pt_2d)))
        bit  = np.array([np.array(Img_mask)[m][ np.clip(pt_2d[m,0], 0, (img_sh[m][0]-1 + 0.9)), np.clip(pt_2d[m,1], 0, (img_sh[m][1]-1 + 0.9)) ] for m in indx])
        clr  = np.array([np.array(Img)[m][ np.clip(pt_2d[m,0], 0, (img_sh[m][0]-1 + 0.9)), np.clip(pt_2d[m,1], 0, (img_sh[m][1]-1 + 0.9)) ] for m in indx])
        clr  = clr[np.where(np.array(bit) == 1)]
        return clr

"""
input : (이미지갯수, RGB 3채널) --> 여기서 이미지의 갯수는 그리드를 통과하는 이미지의 갯수, RGB 그리드를 통과하는 픽셀의 값
"""
def pick_color_map(colors):
    
    threshold = 5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    color_array = np.array(colors)[:,0:3].reshape(-1,1,3) # (N x image_w x image_h x 3) --> (N x (image_h*image_w) x 1 x 3)

    # color_differences : warping 된 여러 이미지의 같은 픽셀에서의 칼라값의 차이
    # color_distances : color_differences의 합의 제곱근 (거리 계산)
    color_differences = color_array[:, np.newaxis] - color_array
    color_distances = np.sqrt(np.sum(abs(color_differences) , axis=-1))

    # 각 이미지의 이웃의 수 : 이미지의 각 픽셀에서 color_distances가 Threshold를 넘지 않는 이미지의 갯수
    num_neighbors = np.sum(color_distances <= threshold, axis=-1) - 1
    # 모든 픽셀에 몇장이 겹쳤는지 더해서 가장 많이 겹친 이미지 
    most_common_color_index = np.argmax(np.sum(num_neighbors, axis=0))
    # 가장 많이 겹친 이미지의 masking 된 color 값
    most_common_neighbors = color_array[np.where(num_neighbors[:, most_common_color_index] == 0)]
    # 전체 이미지에서 가장 많이 겹친 이미지의 masking된 color의 평균
    most_common_color_mean = np.mean(most_common_neighbors, axis=0)[0]/255.0
    # print(colors.shape, color_array.shape, most_common_color_mean.shape)

    return most_common_color_mean

def save_camera_txt(R_matrix, T_matrix, K_matrix, file_path):
        with open(file_path, 'a') as f:
             f.write('entrinsic\n')
             f.write(str(R_matrix[0,0]) + " " + str(R_matrix[0,1])  + " " +  str(R_matrix[0,2]) + " " + str(T_matrix[0,0]) + " \n" )
             f.write(str(R_matrix[1,0]) + " " + str(R_matrix[1,1])  + " " +  str(R_matrix[1,2]) + " " + str(T_matrix[1,0]) + " \n" )
             f.write(str(R_matrix[2,0]) + " " + str(R_matrix[2,1])  + " " +  str(R_matrix[2,2]) + " " + str(T_matrix[2,0]) + " \n" )
             f.write(str(0) + " " + str(0)  + " " +  str(0) + " " + str(1) + "\n" )
             f.write('intrinsic\n')
             f.write(str(K_matrix[0,0]) + " " + str(K_matrix[0,1])  + " " +  str(K_matrix[0,2])  + "\n" )
             f.write(str(K_matrix[1,0]) + " " + str(K_matrix[1,1])  + " " +  str(K_matrix[1,2])  + "\n" )
             f.write(str(K_matrix[2,0]) + " " + str(K_matrix[2,1])  + " " +  str(K_matrix[2,2])  + "\n" )
    