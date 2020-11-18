import torch as T
from torch.nn.functional import math

def affine_2d(translation=None, rotation=0, scale=None, shear=None):
    if translation is None:
        translation = [0,0]
    if scale is None:
        scale = [1,1]
    if shear is None:
        shear=[0,0]

    transf = T.zeros((3,3))
    c = T.cos(T.Tensor([rotation]))
    s = T.sin(T.Tensor([rotation]))
    # rotation
    transf[:2, :2] = T.Tensor([[c, -s],[s, c]])
    # scaling
    transf[:2, :2] = T.matmul(T.Tensor([[scale[0], 0],[0, scale[1]]]), transf[:2, :2])
    # shearing
    transf[:2, :2] = T.matmul(T.Tensor([[1, shear[0]],[shear[1], 1]]), transf[:2, :2])

    transf[:, -1] = T.Tensor([*translation, 1])
    return transf

def affine_3d(translation=None, rotation=None, scale=None, shear=None):
    '''
    Generates a 3D affine matrix (4x4) according to input parameters. Default values are identity.
    Order of operations, right-most applied first
        Shz * Shy * Shx * Scale * Rz * Ry * Rx
    "Sha" is Shear on dim "a", Ra is Rotation on axis "a"
    Parameters
    ----------
    translation : list
        3x1 vector indicating X,Y,Z translation in voxels.
    rotation : list
        3x1 vector indicating rotations along X,Y,Z axes.
    scale : list
        3x1 vector indicating scales along X,Y,Z axes.
    shear : list
        3x2 vector, indicating shear in one dimension to the other (in order X,Y,Z):
        [[Sh_xy, Sh_xz], [Sh_yx, Sh_yz], [Sh_zx, Sh_zy]]
    Returns
    -------
    Tensor
        4x4 Tensor
    '''
    if translation is None:
        translation=[0,0,0]
    if rotation is None:
        rotation = T.zeros((3,1))
    elif not isinstance(rotation, T.Tensor):
        rotation = T.Tensor(rotation)
    if scale is None:
        scale = T.ones((3,1))
    if shear is None:
        shear = [[0,0],[0,0],[0,0]]
    transf = T.zeros((4,4))
    rc = T.cos(rotation)
    rs = T.sin(rotation)

    # rotation
    rot_x = T.Tensor([[1,0,0],[0, rc[0], -rs[0]], [0, rs[0], rc[0]]])
    rot_y = T.Tensor([[rc[1], 0, rs[1]],[0,1,0], [-rs[1], 0, rc[1]]])
    rot_z = T.Tensor([[rc[2], -rs[2],0], [rs[2], rc[2], 0], [0,0,1]])
    transf[:3, :3] = T.matmul( rot_z, T.matmul(rot_y, rot_x))

    # scale
    scal = T.zeros((3,3))
    scal[0,0] = scale[0]
    scal[1,1] = scale[1]
    scal[2,2] = scale[2]
    transf[:3,:3] = T.matmul(scal, transf[:3,:3])

    # shear
    shx = T.eye(3)
    shx[0,1] = shear[0][0]
    shx[0,2] = shear[0][1]

    shy =T.eye(3)
    shy[1,0] = shear[1][0]
    shy[1,2] = shear[1][1]

    shz = T.eye(3)
    shz[2,0] = shear[2][0]
    shz[2,1] = shear[2][1]
    sh = T.matmul(shz, T.matmul(shy, shx))

    transf[:3, :3] = T.matmul(sh, transf[:3,:3])
    transf[:, -1] = T.Tensor([*translation, 1])
    return transf

def random_affine_3d(translation_limits=None, rotation_limits=None, scale_limits=None,
                     shear_limits=None, return_param=False):
    '''
    Generates an affine matrix specified the random affine parameters. Parameters are sampled uniformly in the range.
    Parameters
    ----------
    translation_limits : Tensor
        Optional. 3x2 Tensor specifying (min, max) translations in X,Y,Z directions. Defaults to 0.
    rotation_limits : Tensor
        Optional. 3x2 Tensor specifying (min, max) rotations along X,Y,Z axes. Defaults to 0.
    scale_limits : Tensor
        Optional. 3x2 Tensor specifying (min, max) scaling along X,Y,Z directions. Defaults to 1.
    shear_limits : Tensor
        Optional. 6x2 Tensor specifying (min, max) for xy, xz, yx, yz, zx, zy. Defaults to 0.
    return_param : bool
        If True, will return the random set of parameters instead of the affine.
    Returns
    -------
    Tensor
        4x4 affine matrix.
    '''
    if translation_limits is None:
        translation_limits = T.zeros((3,2))
        # translation_limits[:,0] = -15
        # translation_limits[:,1] = 15
    if rotation_limits is None:
        rotation_limits = T.zeros((3,2))
        # rotation_limits[:,0] = -math.pi/5
        # rotation_limits[:,1] = math.pi/5
    if(scale_limits is None):
        scale_limits = T.zeros((3,2))
        scale_limits[:,0] = 1
        scale_limits[:,1] = 1
        # scale_limits[:,0] = 0.95
        # scale_limits[:,1] = 1.05
    if(shear_limits is None):
        shear_limits = T.zeros((6,2))
        # shear_limits[:,0] = -0.02
        # shear_limits[:,1] = 0.02

    translation = T.rand((3)) * (translation_limits[:,1] - translation_limits[:,0]) + translation_limits[:,0]
    rotation = T.rand((3)) * (rotation_limits[:,1] - rotation_limits[:,0]) + rotation_limits[:,0]
    scale = T.rand((3)) * (scale_limits[:,1] - scale_limits[:,0]) + scale_limits[:,0]
    shear = T.rand((6)) * (shear_limits[:, 1] - shear_limits[:, 0]) + shear_limits[:, 0]
    shear = shear.view(3,2)

    if return_param:
        return translation, rotation, scale, shear
    else:
        return affine_3d(translation=translation, rotation=rotation, scale=scale, shear=shear)