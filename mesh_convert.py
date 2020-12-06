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
from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model

def convert_mesh(mesh, corr_setup):
    v = np.vstack((mesh.v, np.zeros_like(mesh.v)))
    return Mesh(corr_setup['mtx'].dot(v), corr_setup['f_out'])

def convert_BFM_mesh_to_FLAME(FLAME_model_fname, BFM_mesh_fname, FLAME_out_fname):
    '''
    Convert Basel Face Model mesh to a FLAME mesh
    \param FLAME_model_fname        path of the FLAME model
    \param BFM_mesh_fname           path of the BFM mesh to be converted
    \param FLAME_out_fname          path of the output file
    '''

    # Regularizer weights for jaw pose (i.e. opening of mouth), shape, and facial expression.
    # Increase regularization in case of implausible output meshes. 
    w_pose = 1e-4
    w_shape = 1e-3
    w_exp = 1e-4

    if not os.path.exists(os.path.dirname(FLAME_out_fname)):
        os.makedirs(os.path.dirname(FLAME_out_fname))

    if not os.path.exists(BFM_mesh_fname):
        print('BFM mesh not found %s' % BFM_mesh_fname)
        return
    BFM_mesh = Mesh(filename=BFM_mesh_fname)

    if not os.path.exists(FLAME_model_fname):
        print('FLAME model not found %s' % FLAME_model_fname)
        return
    model = load_model(FLAME_model_fname)

    if not os.path.exists('./data/BFM_to_FLAME_corr.npz'):
        print('Cached mapping not found')
        return
    cached_data = np.load('./data/BFM_to_FLAME_corr.npz', allow_pickle=True, encoding="latin1")

    BFM2017_corr = cached_data['BFM2017_corr'].item()
    BFM2009_corr = cached_data['BFM2009_corr'].item()
    BFM2009_cropped_corr = cached_data['BFM2009_cropped_corr'].item()

    if (2*BFM_mesh.v.shape[0] == BFM2017_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2017_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2017_corr)
    elif (2*BFM_mesh.v.shape[0] == BFM2009_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2009_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2009_corr)
    elif (2*BFM_mesh.v.shape[0] == BFM2009_cropped_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2009_cropped_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2009_cropped_corr)
    else:
        print('Conversion failed - input mesh does not match any setup')
        return

    FLAME_mask_ids = cached_data['FLAME_mask_ids']

    scale = ch.ones(1)
    v_target = scale*ch.array(conv_mesh.v)
    dist = v_target[FLAME_mask_ids]-model[FLAME_mask_ids]
    pose_reg = model.pose[3:]
    shape_reg = model.betas[:300]
    exp_reg = model.betas[300:]
    obj = {'dist': dist, 'pose_reg': w_pose*pose_reg, 'shape_reg': w_shape*shape_reg, 'exp_reg': w_exp*exp_reg}
    ch.minimize(obj, x0=[scale, model.trans, model.pose[:3]])
    ch.minimize(obj, x0=[scale, model.trans, model.pose[np.hstack((np.arange(3), np.arange(6,9)))], model.betas])

    v_out = model.r/scale.r
    Mesh(v_out, model.f).write_obj(FLAME_out_fname)


def main():
    # FLAME model filename (download from flame.is.tue.mpg.de)
    FLAME_model_fname = './model/generic_model.pkl'
    # BFM mesh to be converted 
    BFM_mesh_fname = './data/134212_1.obj'
    # Output FLAME mesh filename
    FLAME_out_fname = './output/134212_1_FLAME.obj'
    convert_BFM_mesh_to_FLAME(FLAME_model_fname, BFM_mesh_fname, FLAME_out_fname)

if __name__ == '__main__':
    print('Conversion started......')
    main()
    print('Conversion finished')