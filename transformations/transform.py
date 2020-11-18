import torch as T
import numpy as np
from pyt_transformations import interpolation as interp
from pyt_transformations import matrices
from matplotlib import pyplot as plt
import nibabel as nb

def affine(dat: T.Tensor, mat_affine: T.Tensor, device: str=None, interpolation: str='nearest',border_mode: str='zero'):
    '''
    Applies an affine transformation defined by the 4x4 matrix mat_affine to data
    Parameters
    ----------
    dat : Tensor
        Data to be transformed. Expects 3D.
    mat_affine : Tensor
        4x4 matrix specifying an affine transformation. Can be generated from parameters using 'matrices.py'
    device : str
        Optional. Specifies device to perform transformation. Defaults to the same device that 'dat' is on.
    interpolation : str
        Optional. Type of interpolation to be done (defined in 'interpolation.py'). Valid values: ['nearest', 'linear']
    border_mode : str
        Optional. Type of border mode. Valid values: ['zero']

    Returns
    -------
    Tensor
        'dat' that has undergone the specified transformation.
    '''
    # Note: 'mat_affine' is more about the _intent_ of this function. We don't verify or enforce affine-ness.

    ndim = len(dat.shape)
    if(device is None):
        device = dat.device
    inv_mat = T.inverse(mat_affine).to(device)
    midpoint = [T.floor(T.Tensor([i/2])).int() for i in dat.shape]  # origin is at the centre of the image
    slices = [slice(-midpoint[i], midpoint[i]) for i in range(ndim)]
    dat_loc = T.from_numpy(np.mgrid[slices]).float().to(device)  # initial grid
    dat_flat = dat_loc.view(ndim, -1)
    new_pos = T.matmul(inv_mat, T.cat([dat_flat,T.ones((1, dat_flat.shape[-1])).to(device)], 0))  # sampling grid

    for i in range(ndim):
        new_pos[i, ...] += midpoint[i].float().to(device)
    if(interpolation == 'nearest'):
        return interp.interp_nearest(dat, new_pos[:-1,...].view(ndim, *dat_loc.shape[1:]), border_mode=border_mode)
    elif(interpolation == 'linear'):
        return interp.interp_linear(dat, new_pos[:-1,...].view(ndim, *dat_loc.shape[1:]), border_mode=border_mode)
    else:
        raise NotImplementedError(f'interpolation type {interpolation} has not been implemented')