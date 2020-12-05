'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import os
import cv2
import h5py
import numpy as np
import chumpy as ch

def load_BFM_2017(fname):
    '''
    Loads BFM 2017 in h5 file format and returns a chumpy object and all model parameters
    '''

    with h5py.File(fname, 'r') as f:
        shape_mean = f['shape']['model']['mean'][:]
        shape_pcaBasis = f['shape']['model']['pcaBasis'][:]
        shape_pcaVariance = f['shape']['model']['pcaVariance'][:]

        expression_mean = f['expression']['model']['mean'][:]
        expression_pcaBasis = f['expression']['model']['pcaBasis'][:]
        expression_pcaVariance = f['expression']['model']['pcaVariance'][:]

        color_mean = f['color']['model']['mean'][:]
        color_pcaBasis = f['color']['model']['pcaBasis'][:]
        color_pcaVariance = f['color']['model']['pcaVariance'][:]

        shape_coeffs = ch.zeros(shape_pcaBasis.shape[1])
        exp_coeffs = ch.zeros(expression_pcaBasis.shape[1])
        color_coeffs = ch.zeros(color_pcaBasis.shape[1])

        sc = ch.diag(np.sqrt(shape_pcaVariance)).dot(shape_coeffs)
        ec = ch.diag(np.sqrt(expression_pcaVariance)).dot(exp_coeffs)
        cc = ch.diag(np.sqrt(color_pcaVariance)).dot(color_coeffs)
        v_bfm = ch.array(shape_mean).reshape(-1,3) + ch.array(shape_pcaBasis).dot(sc).reshape(-1,3) + \
                ch.array(expression_mean).reshape(-1, 3) + ch.array(expression_pcaBasis).dot(ec).reshape(-1, 3)
        c_bfm = ch.array(color_mean).reshape(-1, 3)  + ch.array(color_pcaBasis).dot(cc).reshape(-1, 3)
        return {'verts': v_bfm, 'color': c_bfm, 'shape_coeffs': shape_coeffs, 'exp_coeffs': exp_coeffs, 'color_coeffs': color_coeffs}

# def cache_data():
#     from psbody.mesh import Mesh

#     aligned_BFM_template = Mesh(filename='./data/BFM_aligned.obj')
#     FLAME_uv_template = Mesh(filename='./data/FLAME_uv_template.obj')

#     def barycentric_coords(a, b, c, q):
#         '''Compute Barycentric coordinates of q projected on the triangle spanned by the vertices a, b, and c'''
#         v1 = b-a
#         v2 = c-a
#         n = np.cross(v1, v2)
#         n_dot_n = np.sum(n*n, axis=-1)
#         w = q - a

#         gamma = np.sum(np.cross(v1, w)*n, axis=-1)/n_dot_n
#         beta = np.sum(np.cross(w, v2)*n, axis=-1)/n_dot_n
#         alpha = 1.0-beta-gamma
#         return np.vstack((alpha, beta, gamma)).T

#     w, h = 512, 512

#     # Compute uv grid with a 2D vertex for each pixel
#     x_coords = np.reshape(np.array([[i+0.5]*h for i in range(w)]), (-1,))
#     y_coords = np.reshape(np.tile(np.array([i+0.5 for i in range(h)]), reps=w), (-1,))
#     grid = np.vstack((x_coords, y_coords)).T

#     vt = np.zeros_like(FLAME_uv_template.vt)
#     vt[:,0] = w*FLAME_uv_template.vt[:,0]
#     vt[:,1] = h-h*FLAME_uv_template.vt[:,1]

#     uv_coord_mesh_3d = Mesh(np.hstack((vt, np.zeros((vt.shape[0], 1)))), FLAME_uv_template.ft)
#     grid_3d = np.hstack((grid, np.zeros((grid.shape[0], 1))))
#     nearest_faces, _, _ = uv_coord_mesh_3d.compute_aabb_tree().nearest(grid_3d, True)
#     nearest_faces = np.squeeze(nearest_faces)

#     # Coordinates of the corners of the closest mesh triangle for each grid point 
#     v1 = uv_coord_mesh_3d.v[FLAME_uv_template.ft[nearest_faces][:, 0]]
#     v2 = uv_coord_mesh_3d.v[FLAME_uv_template.ft[nearest_faces][:, 1]]
#     v3 = uv_coord_mesh_3d.v[FLAME_uv_template.ft[nearest_faces][:, 2]]
#     b_coords = barycentric_coords(v1, v2, v3, grid_3d)

#     # Filter all points that are not inside a triangle in the uv map
#     wrong_b_coords = np.union1d(np.where(b_coords.flatten() < 0)[0], np.where(b_coords.flatten() > 1)[0])
#     wrong_ids = np.unique((np.floor(wrong_b_coords / 3.0)).astype(int))

#     valid_pixel_ids = np.setdiff1d(np.arange(w*h), wrong_ids)
#     valid_pixel_3d_faces = FLAME_uv_template.f[nearest_faces[valid_pixel_ids], :]
#     valid_pixel_b_coords = b_coords[valid_pixel_ids, :]

#     pixel_3d_points = FLAME_uv_template.v[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
#                       FLAME_uv_template.v[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
#                       FLAME_uv_template.v[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]

#     nearest_BFM_faces, _, _ = aligned_BFM_template.compute_aabb_tree().nearest(pixel_3d_points, True)
#     nearest_BFM_faces = np.squeeze(nearest_BFM_faces)

#     v1_BFM = aligned_BFM_template.v[aligned_BFM_template.f[nearest_BFM_faces][:, 0]]
#     v2_BFM = aligned_BFM_template.v[aligned_BFM_template.f[nearest_BFM_faces][:, 1]]
#     v3_BFM = aligned_BFM_template.v[aligned_BFM_template.f[nearest_BFM_faces][:, 2]]
#     b_coords_bfm = barycentric_coords(v1_BFM, v2_BFM, v3_BFM, pixel_3d_points)

#     wrong_b_coords_BFM = np.union1d(np.where(b_coords_bfm.flatten() < -0.5)[0], np.where(b_coords_bfm.flatten() > 1.5)[0])
#     wrong_ids_BFM = np.unique((np.floor(wrong_b_coords_BFM / 3.0)).astype(int))
#     valid_BFM_ids = np.setdiff1d(np.arange(b_coords_bfm.shape[0]), wrong_ids_BFM)

#     valid_BFM_faces = aligned_BFM_template.f[nearest_BFM_faces[valid_BFM_ids]]
#     valid_BFM_b_coords = b_coords_bfm[valid_BFM_ids, :]
#     ids = valid_pixel_ids[valid_BFM_ids]

#     cached_data = {'w': w, 'h': h, 'ids': ids, 'x_coords': x_coords, 'y_coords': y_coords, 'valid_BFM_faces': valid_BFM_faces, 'valid_BFM_b_coords': valid_BFM_b_coords}
#     return cached_data

def transfer_BFM_textures_to_FLAME_uv():
    '''
    Convert Basel Face Model vertex colors to FLAME texture space
    '''

    if not os.path.exists('./data/cached_data.npy'):
        print('Cached data not found')
        return

    cached_data = np.load('./data/cached_data.npy', allow_pickle=True, encoding='latin1').item()

    w, h = cached_data['w'], cached_data['h']
    x_coords, y_coords, ids = cached_data['x_coords'], cached_data['y_coords'], cached_data['ids']
    valid_BFM_faces, valid_BFM_b_coords = cached_data['valid_BFM_faces'], cached_data['valid_BFM_b_coords']

    bfm_model = load_BFM_2017('./model/model2017-1_bfm_nomouth.h5')
    color = bfm_model['color']
    color_coeffs = bfm_model['color_coeffs']
    num_color_components = color_coeffs.shape[0]

    # Get cached inpainted region for outside face area
    inpaint = False
    if os.path.exists('./data/mask_inpainting.npz'):
        inpainted_mask = np.load('./data/mask_inpainting.npz')   
        if ('MU' in inpainted_mask) and ('PC' in inpainted_mask) and (inpainted_mask['PC'].shape[1] == num_color_components):
            mask_ids = np.setdiff1d(np.arange(w*h), ids)
            low_res = int(np.sqrt(inpainted_mask['MU'].shape[0]/3))
            inpaint = True
    if not inpaint:
        print('Cached inpainting not found, outside face area will not be inpainted')

    tmp_coeffs = np.zeros_like(color_coeffs.r)
    color_coeffs[:] = tmp_coeffs
    bfm_verts_color = color.r

    mean_point_color = bfm_verts_color[valid_BFM_faces[:, 0], :] * valid_BFM_b_coords[:, 0][:, np.newaxis] + \
                       bfm_verts_color[valid_BFM_faces[:, 1], :] * valid_BFM_b_coords[:, 1][:, np.newaxis] + \
                       bfm_verts_color[valid_BFM_faces[:, 2], :] * valid_BFM_b_coords[:, 2][:, np.newaxis]

    mean_img = np.zeros((h, w, 3))
    if inpaint:
        inpainted_mean = cv2.resize(inpainted_mask['MU'].reshape((low_res, low_res, 3)), (h, w))
        mean_img[y_coords[mask_ids].astype(int), x_coords[mask_ids].astype(int), :3] = inpainted_mean[y_coords[mask_ids].astype(int), x_coords[mask_ids].astype(int), :3]
    mean_img[y_coords[ids].astype(int), x_coords[ids].astype(int), ::-1] = mean_point_color
    mean = np.reshape(mean_img, [-1,])

    basis = np.zeros((h*w*3, num_color_components))
    for i in range(num_color_components):
        tmp_coeffs = np.zeros_like(color_coeffs.r)
        tmp_coeffs[i] = 1.0
        color_coeffs[:] = tmp_coeffs
        bfm_verts_color = color.r

        point_color =     bfm_verts_color[valid_BFM_faces[:, 0], :] * valid_BFM_b_coords[:, 0][:, np.newaxis] + \
                          bfm_verts_color[valid_BFM_faces[:, 1], :] * valid_BFM_b_coords[:, 1][:, np.newaxis] + \
                          bfm_verts_color[valid_BFM_faces[:, 2], :] * valid_BFM_b_coords[:, 2][:, np.newaxis]
        
        img = np.zeros((h, w, 3))
        if inpaint:
            inpainted_PC = cv2.resize(inpainted_mask['PC'][:,i].reshape((low_res, low_res, 3)), (h, w))
            img[y_coords[mask_ids].astype(int), x_coords[mask_ids].astype(int), :3] = inpainted_PC[y_coords[mask_ids].astype(int), x_coords[mask_ids].astype(int), :3]
        img[y_coords[ids].astype(int), x_coords[ids].astype(int), ::-1] = point_color-mean_point_color
        basis[:, i] = np.reshape(img, [-1,])

    out_folder = './output'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    np.savez(os.path.join(out_folder, 'FLAME_albedo_from_BFM.npz'), MU=mean, PC=basis)

def main():
    transfer_BFM_textures_to_FLAME_uv()

if __name__ == '__main__':
    # cached_data = cache_data()
    # np.save('./data/cached_data.npy', cached_data)

    print('Conversion started......')
    main()
    print('Conversion finished')