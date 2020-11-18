import torch as T

def interp_nearest(data, sample_locations, border_mode='zero'):
    '''
    Performs nearest-neighbour interpolation
    Parameters
    ----------
    data : Tensor
        data from which to sample
    sample_locations : Tensor
        locations to sample
    border_mode : str
        How to handle out-of-border data.
    Returns
    -------
    Tensor
        Interpolated data.
    '''
    # data: X x Y x Z
    # sample_locations: [ndim x sample_points]
    ndim = len(data.shape)
    # voxel coordinates from which to sample
    sample_nn = T.round(sample_locations).long()
    device = data.device
    # exceeds image limits?
    #print(time() - start)
    out_of_bounds_0 = T.stack([sample_nn[i, ...] >= T.Tensor([data.shape[i]]).long().to(device) for i in range(ndim)], dim=0)
    out_of_bounds_1 = T.stack([sample_nn[i, ...] < T.zeros(1).long().to(device) for i in range(ndim)], dim=0)
    #print(time() - start)
    out_of_bounds = out_of_bounds_0 + out_of_bounds_1
    out_of_bounds = out_of_bounds != 0
    sample_nn[out_of_bounds] = 0
    #print(time() - start)
    #sample_nn[out_of_bounds.expand(ndim, -1)] = 0

    # this generalizes well to N dimensions, but takes 120x as long (~24s) in the typical 3d case
    # TODO: Use sub2ind to get around tuple indexing
    # tuple_ind = [tuple(sample_nn[i, ...]) for i in range(ndim)]
    # new_image = data[tuple_ind]

    # works fast, but assumes 3D data
    new_image = data[sample_nn[0,...], sample_nn[1,...], sample_nn[2,...]]

    #print(time() - start)
    if(border_mode == 'zero'):
        new_image[out_of_bounds[0,...]] = 0
    else:
        raise NotImplementedError(f'border_mode = {border_mode} has not been implemented')
    return new_image.view(*sample_locations.shape[1:])


def interp_linear(data, sample_locations, border_mode='zero'):
    '''
    Performs linear interpolation.
    Parameters
    ----------
    data : Tensor
        data from which to sample
    sample_locations : Tensor
        locations to sample
    border_mode : str
        How to handle out-of-border data.
    Returns
    -------
    Tensor
        Interpolated data.
    '''
    # At every sample location, get surrounding 2^ndim points in data and take weighted mean
    device = data.device
    data_pad = T.zeros((data.shape[0]+2, data.shape[1]+2, data.shape[2]+2), device=device)
    data_pad[1:-1, 1:-1, 1:-1] = data
    ndim = len(data.shape)
    sample_shape = sample_locations.shape
    sample_reshape = (sample_locations + T.ones(sample_shape, device=device)).view(3, -1)

    # voxel coordinates from which to sample
    #sample_nn = T.round(sample_locations).long()
    # print('data_shape: ' + str(data.shape))
    # Assume 3d
    fx = sample_reshape[0,:] - T.floor(sample_reshape[0,:])
    fy = sample_reshape[1, :] - T.floor(sample_reshape[1, :])
    fz = sample_reshape[2, :] - T.floor(sample_reshape[2, :])
    im = T.zeros(sample_reshape.shape[1], device=device)
    # im = T.zeros((5,5,5))
    for z_ind in range(2):
        fz = 1-fz
        z_func = T.floor if z_ind == 0 else T.ceil
        for y_ind in range(2):
            fy = 1-fy
            y_func = T.floor if y_ind == 0 else T.ceil
            for x_ind in range(2):
                fx = 1-fx
                x_func = T.floor if x_ind == 0 else T.ceil
                x_indices = T.clamp( x_func(sample_reshape[0,:]), 0, data.shape[0]-1).long()
                y_indices = T.clamp( y_func(sample_reshape[1,:]), 0, data.shape[1]-1).long()
                z_indices = T.clamp( z_func(sample_reshape[2,:]), 0, data.shape[2]-1).long()
                im += fz*fy*fx * data_pad[x_indices, y_indices, z_indices]

    return im.view(*sample_shape[1:])
# def _sub2ind(subs, dat_shape):
#     ndim = subs.shape[0]
#
#     shp = [1]
#     shp += [i for i in dat_shape[:0:-1]]
#     shp = T.cumprod(T.Tensor(shp),0)
#
#     return T.matmul(shp, subs)
