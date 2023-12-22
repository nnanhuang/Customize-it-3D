import open3d as o3d
import numpy as np
import time
import math
import itertools
import torch
import torch.nn.functional as F
import tqdm
import imageio
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer import AlphaCompositor
from pytorch3d.renderer.points import rasterize_points
import cv2
import os

def mesh2v(path_obj):
    mesh = o3d.io.read_triangle_mesh(path_obj)
    
    start_time = time.time() 
    # pcd = mesh.sample_points_poisson_disk(number_of_points=500000,pcl=pcd_ori)
    print(f'[INFO] sampling points on mesh...')
    pcd = mesh.sample_points_poisson_disk(number_of_points=700000,init_factor=10)
    end_time = time.time() 
    
    elapsed_time = end_time - start_time
    print(f"using time: {elapsed_time} s")
    
    v = np.asarray(pcd.points)
    
    return v

def removel_mesh(mesh):
    # Create bounding box:
    bounds = [[-0.5, 0.5], [-0.4, 0.4], [-math.inf,math.inf]]   # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

    # Crop the point cloud using the bounding box:
    mesh_croped = mesh.crop(bounding_box)
    
    return mesh_croped

def removel_pcd(pcd,x1,x2,y1,y2,z1,z2):
    # Create bounding box:
    bounds = [[x1, x2], [y1, y2], [z1,z2]]   # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

    # Crop the point cloud using the bounding box:
    pcd_croped = pcd.crop(bounding_box)
    
    return pcd_croped

def get_v_color(v,K,w2c,gt_rgb,H):
    xy,_ = project(v ,K, w2c[:3,:4])
    gt_rgb = torch.Tensor(gt_rgb[None,...]).permute(0,3,1,2)
    xy =torch.Tensor(xy[None,None,...])/H*2.-1.
    v_color=F.grid_sample(gt_rgb,xy)
    v_color = v_color.squeeze().permute(1,0).cpu().numpy()
    
    return v_color

def project(xyz, K, RT):
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy,xyz[:,2:]
 
def segment_pcd(v):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v[:,0:3])

    front_pcd = removel_pcd(pcd,-math.inf,math.inf,-math.inf,math.inf,0,math.inf) 
    back_pcd = removel_pcd(pcd,-math.inf,math.inf,-math.inf,math.inf,-math.inf,0) 
    
    v1 = np.asarray(front_pcd.points)
    v2 = np.asarray(back_pcd.points)

    return v1, v2

def load_depth(file_name):
    import cv2
    D = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if not D is None and len(D.shape) > 2:
        D = D[:,:,0]
    return    D

def z_buffer_cano(vertices, world2cam_pose_cano, H, W, K,gt_mask=None):
    # world to camera
    xy1,xyz = project(vertices, K, world2cam_pose_cano[:3,:4])
    xy1 = np.round(xy1).astype(np.int32)
    xy1_mask =np.logical_and(np.logical_and((xy1[:,0]>=0),(xy1[:,0]<=W-1)),np.logical_and((xy1[:,1]>=0),(xy1[:,1]<=H-1)))

    # zbuffer, find visible point
    img1 = np.zeros((H,W)) 
    zbuff = np.zeros((H,W,2)) 
    maskout =[]
    colorout = []
    for i in np.arange(xy1.shape[0]): 
        if xy1_mask[i]: 
            if img1[xy1[i, 1], xy1[i, 0]]==0:
                img1[xy1[i, 1], xy1[i, 0]] = 1 
                zbuff[xy1[i,1], xy1[i,0],0] = xyz[i,0]
                zbuff[xy1[i,1], xy1[i,0],1] = i
            elif xyz[i,0]<zbuff[xy1[i,1],xy1[i,0,],0]: 
                zbuff[xy1[i,1],xy1[i,0],0] = xyz[i,0] 
                zbuff[xy1[i,1], xy1[i,0],1] = i
            else:
                continue 
    for i in np.arange(xy1.shape[0]):
        if xy1_mask[i] and gt_mask[xy1[i,1],xy1[i,0]]: 
            if (xyz[i,0]-zbuff[xy1[i,1],xy1[i,0,],0])> (1./H): 
                #  continue
                maskout.append(vertices[int(i):int(i)+1,:]) 
                colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
            else:
                maskout.append(vertices[int(i):int(i)+1,:]) 
                colorout.append(np.array([1,1,1],dtype=np.float32).reshape((1,3)))
        else:
            maskout.append(vertices[int(i):int(i)+1,:]) 
            colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
    maskout = np.concatenate(maskout,axis=0)
    colorout = np.concatenate(colorout,axis=0)
    vertices = maskout
    colors = colorout
    mask =((colors[:,0]+colors[:,1]+colors[:,2])!=0)
    
    return mask

def z_buffer_final(vertices, world2cam_pose_cano, H, W, K,gt_mask=None):
    # world to camera
    xy1,xyz = project(vertices, K, world2cam_pose_cano[:3,:4])
    xy1 = np.round(xy1).astype(np.int32)

    # zbuffer, find visible point
    maskout =[]
    colorout = []
 
    for i in np.arange(xy1.shape[0]): 
        if gt_mask[xy1[i,1],xy1[i,0]]: 
            maskout.append(vertices[int(i):int(i)+1,:]) 
            colorout.append(np.array([1,1,1],dtype=np.float32).reshape((1,3)))
        else:
            maskout.append(vertices[int(i):int(i)+1,:]) 
            colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
    maskout = np.concatenate(maskout,axis=0)
    colorout = np.concatenate(colorout,axis=0)
    vertices = maskout
    colors = colorout
    mask =((colors[:,0]+colors[:,1]+colors[:,2])!=0)
    
    return mask

def z_buffer(vertices, world2cam_pose_cano, H, W, K):
    # world to camera
    xy1,xyz = project(vertices, K, world2cam_pose_cano[:3,:4])
    xy1 = np.round(xy1).astype(np.int32)
    xy1_mask =np.logical_and(np.logical_and((xy1[:,0]>=0),(xy1[:,0]<=W-1)),np.logical_and((xy1[:,1]>=0),(xy1[:,1]<=H-1)))

    # zbuffer, find visible point
    img1 = np.zeros((H,W)) 
    zbuff = np.zeros((H,W,2)) 
    maskout =[]
    colorout = []
    for i in np.arange(xy1.shape[0]): 
        if xy1_mask[i]: 
            if img1[xy1[i, 1], xy1[i, 0]]==0: 
                img1[xy1[i, 1], xy1[i, 0]] = 1 
                
                zbuff[xy1[i,1], xy1[i,0],0] = xyz[i,0]
                zbuff[xy1[i,1], xy1[i,0],1] = i
            elif xyz[i,0]<zbuff[xy1[i,1],xy1[i,0,],0]: 
                zbuff[xy1[i,1],xy1[i,0],0] = xyz[i,0] 
                zbuff[xy1[i,1], xy1[i,0],1] = i
            else:
                continue 
      
    for i in np.arange(xy1.shape[0]): 
        if xy1_mask[i]: 
            if (xyz[i,0]-zbuff[xy1[i,1],xy1[i,0,],0])> (1./H): 
                maskout.append(vertices[int(i):int(i)+1,:]) 
                colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
            else:
                maskout.append(vertices[int(i):int(i)+1,:]) 
                colorout.append(np.array([1,1,1],dtype=np.float32).reshape((1,3)))
        else:
            maskout.append(vertices[int(i):int(i)+1,:]) 
            colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
    maskout = np.concatenate(maskout,axis=0)
    colorout = np.concatenate(colorout,axis=0)
    vertices = maskout
    colors = colorout
    mask =((colors[:,0]+colors[:,1]+colors[:,2])!=0)
    
    return mask

def mesh_cano_color(v,gt_rgb,c2w_list,K,mask_files,H,W):
    ref_num = (len(c2w_list)-1) // 2
    # ref_num = 0
    gt_mask_path = mask_files[ref_num]
    gt_mask = load_depth(gt_mask_path) / 255.0
    gt_mask = cv2.resize(gt_mask,(W,H))
    c2w = c2w_list[ref_num]
    w2c = np.linalg.inv(c2w)
    
    gt_mask = (gt_mask > 0.8 )

    color_mask = z_buffer_cano(v,w2c,H,W,K,gt_mask)

    v_colored = v[color_mask] 
    
    v_cano = v_colored
    v_color_cano = get_v_color(v_colored,K,w2c,gt_rgb,H)  
    
    v_uncolored = v[~color_mask] 
    
    return v_cano,v_color_cano,v_uncolored

def render_point(points_xyz_org, points_color, H, W, K, world2cam, image_size, radius, ppp, bg_feat=None, acc='alphacomposite'):
    proj_xyz = torch.matmul(points_xyz_org, world2cam[:3, :3].T) + world2cam[:3, 3]

    # Perspective projection
    proj_xyz = torch.matmul(proj_xyz, K.T) 
    proj_xyz[:, 0:2] = proj_xyz[:, 0:2] / proj_xyz[:, 2:]
    proj_xyz[:, 0] = proj_xyz[:, 0] / W * 2 - 1.0
    proj_xyz[:, 1] = proj_xyz[:, 1] / H * 2 - 1.0
    proj_xyz[:, 0] = proj_xyz[:, 0] * -1
    proj_xyz[:, 1] = proj_xyz[:, 1] * -1

    pts3D = Pointclouds(points=[proj_xyz], features=[points_color])
    # print(proj_xyz) # [44410,3], tensor
    points_idx, z_buf, dist = rasterize_points(
        pts3D, image_size, radius, ppp)
    # points = proj_xyz
    # p = points_idx[0,400,400,4] # points_idx: [1,800,800,8], tensor
    # print(points[p])
    dist = 0.1*dist/ pow(radius, 2)
    alphas = (
        (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
        .pow(1)
        .permute(0, 3, 1, 2)
    )

    transformed_src_alphas = compositing.alpha_composite(
        points_idx.permute(0, 3, 1, 2).long(),
        alphas,
        pts3D.features_packed().permute(1,0),
    )
        
    return transformed_src_alphas

def mesh_color_novel(v,rgb_files,c2w_list,K,mask_files,v_cano,H,W,radius,ppp,device,outputdir):

    pbar = tqdm.tqdm(range(c2w_list.shape[0]),desc="coloring point clouds...")
    v_color_list = []
    v_list = []
    ref_num = (len(c2w_list)-1) // 2
    # ref_num = 0
    
    image_size = [H, W]
    v_cano_tensor = torch.tensor(v_cano).to(device).float()
    cano_color = torch.ones_like(v_cano_tensor)
    K_tensor = torch.tensor(K).to(device).float()
    radius = float(radius) / float(image_size[0]) * 2.0

    for i in pbar:
        if i == ref_num:
            continue
        else:
            c2w = c2w_list[i]
            w2c = np.linalg.inv(c2w)
            rgb = imageio.imread(rgb_files[i])/255.
            rgb = cv2.resize(rgb[:,:,:3],(H, W))
            gt_mask_path = mask_files[i]
            gt_mask = load_depth(gt_mask_path) / 255.0
            gt_mask = cv2.resize(gt_mask,(W,H))
            gt_mask = (gt_mask == 1 )
            if i == ref_num :
                mask = gt_mask
            else:
                w2c_tensor = torch.tensor(w2c).to(device)
                novel_cano_mask = render_point(v_cano_tensor, cano_color, H, W, K_tensor, w2c_tensor, image_size, radius, ppp)
                imageio.imwrite(outputdir+'/render_%s_mask.png'%(i),np.array(novel_cano_mask[0].permute(1,2,0).detach().cpu().numpy() * 255,dtype=np.uint8))
                mask = imageio.imread(outputdir+'/render_%s_mask.png'%(i)) / 255 
                os.remove(outputdir+'/render_%s_mask.png'%(i))
                kernel = np.ones(((5,5)), np.uint8) ##15
                mask = cv2.erode(mask,kernel,iterations=1) 
                mask = np.logical_or(np.logical_or((mask[:,:, 0]>0.9),(mask[:,:, 1]>0.9)), (mask[:,:, 2]>0.9))
                mask = np.logical_and(gt_mask, ~mask)
            # imageio.imwrite(outputdir+'/render_%s_mask1.png'%(i),mask*255)
            
            color_mask = z_buffer_cano(v,w2c,H,W,K,mask) 
            
            v_colored = v[color_mask] 
            if v_colored.shape[0] == 1 or v_colored.shape[0] == 0:
                continue
            v_color = get_v_color(v_colored,K,w2c,rgb,H)
            
            v_list.append(v_colored)
            v_color_list.append(v_color)
            
            v = v[~color_mask] # 578748,187404,32489,16955,9309,5027,1999
            
            if v.shape[0] == 1 or v.shape[0] == 0:
                break
        
    all_v = np.concatenate(v_list)
    all_v_color = np.concatenate(v_color_list)
    
    return all_v, all_v_color

def final_front_mask(v,v_color,mask_files,c2w_list,K,H,W,mesh_pcd):
    ref_num = (len(c2w_list)-1) // 2
    # ref_num = 0
    gt_mask_path = mask_files[ref_num]
    gt_mask = load_depth(gt_mask_path) / 255.0
    gt_mask = cv2.resize(gt_mask,(W,H))
    
    if not mesh_pcd:
        kernel = np.ones(((11, 11)), np.uint8) ##11
        gt_mask = cv2.erode(gt_mask,kernel,iterations=2)
    
    c2w = c2w_list[ref_num]
    w2c = np.linalg.inv(c2w) 
    gt_mask = (gt_mask > 0.8 )
    
    final_color_mask = z_buffer_final(v,w2c,H,W,K,gt_mask)
    
    final_v = v[final_color_mask]
    final_v_color = v_color[final_color_mask]
    
    return final_v,final_v_color

def mirror_flip_point_cloud(v):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(v[:,0:3])
    mirrored_points = point_cloud.points
    
    for i in range(len(mirrored_points)):
        mirrored_points[i][0] = -mirrored_points[i][0]
        
    mirrored_point_cloud = o3d.geometry.PointCloud(mirrored_points)
    
    v_mirror = np.asarray(mirrored_point_cloud.points)
    
    return v_mirror
  
if __name__=="__main__":
    v_path = 'results/cat_statue/refine/all_v_m.npy' # numpy
    outputdir = 'results/cat_statue/mesh'

    v = np.load()
    np.save(outputdir + f'/mesh3_1000k_v.npy', v)