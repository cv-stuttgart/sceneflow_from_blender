import OpenEXR
import numpy as np
import flow_IO


THRESH1 = 0.01
THRESH2 = 0.5


def disp1ToFlow(disp1, is_left=True):
    multiplier = -1.0 if is_left else 1.0
    return multiplier * np.dstack((disp1, np.zeros_like(disp1)))


def disp2ToFlow(flow, disp2, is_left=True):
    multiplier = -1.0 if is_left else 1.0
    flow[...,0] += multiplier * disp2
    return flow


def matchmap(flow_fw, flow_bw, factor=1.0):
    flow_fw = flow_fw.astype(np.float64)
    flow_bw = flow_bw.astype(np.float64)
    flow_fw *= factor
    flow_bw *= factor
    flow_bw_flat = flow_bw.reshape(-1,2)

    h, w, _ = flow_fw.shape
    indices = np.indices((h, w))
    indices = np.dstack((indices[1], indices[0]))
    indices = indices.astype(np.float64)

    indices += flow_fw

    h, w = indices.shape[:2]
    nanmap = np.isnan(indices[...,0]) | np.isnan(indices[...,1])
    indices_ = np.round(indices).astype(int)

    outwards_map = (indices_[...,0] < 0) | (indices_[...,1] < 0) | (indices_[...,0] >= w) | (indices_[...,1] >= h)
    outwards_map[nanmap] = False
    indices_[indices_[...,0] < 0, 0] = 0
    indices_[indices_[...,1] < 0, 1] = 0
    indices_[indices_[...,0] >= w, 0] = w - 1
    indices_[indices_[...,1] >= h, 1] = h - 1

    indices_ = indices_[...,1] * w + indices_[...,0]

    warped = flow_bw_flat[indices_]

    l2_sq = ((flow_fw + warped)**2).sum(axis=-1)
    sq_sum = (flow_fw**2).sum(axis=-1) + (warped**2).sum(axis=-1)

    if factor == 2.0:
        l2_sq = np.dstack((l2_sq[::2,::2], l2_sq[::2,1::2], l2_sq[1::2,::2], l2_sq[1::2,1::2]))
        l2_sq = np.nansum(l2_sq, axis=-1) / np.maximum(4 - np.isnan(l2_sq).sum(axis=-1), 1)

        sq_sum = np.dstack((sq_sum[::2,::2], sq_sum[::2,1::2], sq_sum[1::2,::2], sq_sum[1::2,1::2]))
        sq_sum = np.nansum(sq_sum, axis=-1) / np.maximum(4 - np.isnan(sq_sum).sum(axis=-1), 1)

        outwards_map = np.dstack((outwards_map[::2,::2], outwards_map[::2,1::2], outwards_map[1::2,::2], outwards_map[1::2,1::2])).astype(np.float32).mean(axis=-1)
        outwards_map = outwards_map >= 0.5

    result = l2_sq > THRESH1 * sq_sum + THRESH2

    result = result | outwards_map
    return result


def readEXR(filepath):
    exrfile = OpenEXR.InputFile(filepath)

    # Compute the size
    dw = exrfile.header()['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    full_img = np.dstack([np.frombuffer(exrfile.channel(c), dtype=np.float32).reshape((h,w)) for c in ["R", "G", "B"]])
    return full_img


def get_invalid(depth):
    return np.isnan(depth) | (depth>10000)


def get_sky(depth, clip_end):
    return (depth > clip_end * 0.98) & (~get_invalid(depth))


def matmul3D(mat, tensor):
    """compute matrix multiplication mat @ vec for every vec in tensor"""
    return np.einsum('ijk,lk->ijl', tensor, mat)


def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs[:,:,0], Xs[:,:,1], Xs[:,:,2]
    fx, fy, cx, cy = intrinsics

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = np.stack([x, y, d], axis=-1)
    return coords


def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape

    fx, fy, cx, cy = intrinsics

    y, x = np.meshgrid(np.arange(ht), np.arange(wd))

    X = depths * ((x.T - cx) / fx)
    Y = depths * ((y.T - cy) / fy)
    Z = depths

    return np.stack([X, Y, Z], axis=-1)


def invert_transformation_matrix(mat):
    rot_part = mat[:3,:3]
    trans_part = mat[:3,3]
    trans_part_ = - rot_part.T @ trans_part
    result_mat = mat.copy()
    result_mat[:3,:3] = rot_part.T
    result_mat[:3,3] = trans_part_
    return result_mat


def depth_conversion(A_int, width, height):
    A_int_inv = np.linalg.inv(A_int)
    point1 = np.indices((width, height))
    point1 = np.transpose(point1, (1,2,0))

    point1_hom = np.dstack((point1, np.ones((width, height, 1))))
    point_3D = np.einsum('ijk,lk->ijl', point1_hom, A_int_inv)
    return np.linalg.norm(point_3D, axis=-1).T


def get_sceneflow(vec3dpath, extrinsics1, extrinsics2, intrinsics1, intrinsics2, depth, disparity, skymap, baseline_width, factor=1.0):

    vec3d = readEXR(vec3dpath)

    intrinsics1 *= factor
    intrinsics2 *= factor

    rel_matrix = extrinsics2 @ invert_transformation_matrix(extrinsics1)

    vec3d *= -1
    vec3d[:,:,1] *= -1

    # compute optical flow + disparity change
    point1_3D = inv_project(depth, intrinsics1)
    point2_3D = point1_3D + vec3d
    pos1 = project(point1_3D, intrinsics1)
    pos2 = project(point2_3D, intrinsics2)

    pos1[...,2] *= baseline_width * intrinsics1[0]
    pos2[...,2] *= baseline_width * intrinsics2[0]

    flow2d, dispchange = np.split(pos2-pos1, [2], axis=-1)
    dispchange /= factor
    disparity2 = disparity + dispchange[:,:,0]

    # rigidity map
    point1_3D_hom = np.dstack((point1_3D, np.ones((point1_3D.shape[0],point1_3D.shape[1]))))
    point2_3D_hom = matmul3D(rel_matrix, point1_3D_hom)
    point2_3D_RIGID = point2_3D_hom[:,:,:3] / point2_3D_hom[:,:,3, None]

    difference = np.linalg.norm(point2_3D - point2_3D_RIGID, axis=-1)
    rigidmap = difference>1e-3

    # compute optical flow for sky pixels that are at infinity
    point1_3D_sky = inv_project(np.ones_like(depth), intrinsics1)
    point1_3D_sky = np.dstack((point1_3D_sky, np.zeros_like(depth)))
    point2_3D_sky = matmul3D(rel_matrix, point1_3D_sky)
    point2_3D_sky = point2_3D_sky[:,:,:3]
    pos1_sky = project(point1_3D_sky, intrinsics1)
    pos2_sky = project(point2_3D_sky, intrinsics2)
    flow2d_sky, _ = np.split(pos2_sky-pos1_sky, [2], axis=-1)

    # replace optical flow at sky pixels:
    flow2d[skymap] = flow2d_sky[skymap]

    # disparity2 is zero at sky pixels:
    disparity2[skymap] = 0.0

    flow2d /= factor

    detailmap_disp2 = get_detailmap(disparity2)
    detailmap_flow = get_detailmap_flow(flow2d)

    return disparity2, flow2d, rigidmap, detailmap_disp2, detailmap_flow


def get_detailmap(img):
    img = np.dstack((img[::2,::2], img[1::2,::2], img[::2,1::2], img[1::2,1::2]))
    med = np.median(img, axis=-1)
    maxmeddev = (img-med[...,None]).max(axis=-1)

    return maxmeddev>1


def get_detailmap_flow(flow):
    return get_detailmap(flow[...,0]) | get_detailmap(flow[...,1])


def get_depth(path, intrinsics, baseline_width, factor=1, clip_end=10000):

    depth = readEXR(path)[:,:,0]

    invalid_map1 = get_invalid(depth)
    depth[invalid_map1] = np.nan

    skymap = get_sky(depth, clip_end)

    depth[skymap] = np.nan

    A_int = np.asarray([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0,0,1]])
    correction = depth_conversion(A_int, 1920*factor, 1080*factor)
    depth /= correction

    disparity = baseline_width * intrinsics[0] / depth
    disparity /= factor

    # the disparity is zero at sky pixels:
    disparity[skymap] = 0.0

    detailmap_disp1 = get_detailmap(disparity)

    return depth, disparity, skymap, detailmap_disp1



if __name__ == "__main__":
    # reference frame intrinsics
    intrinsics1 = np.loadtxt("intrinsics1.txt")
    # target frame intrinsics
    intrinsics2 = np.loadtxt("intrinsics2.txt")

    # reference frame extrinsics
    extrinsics1 = np.loadtxt("extrinsics1.txt")
    # target frame extrinsics
    extrinsics2 = np.loadtxt("extrinsics2.txt")

    depth_path = "depth.exr"
    vec3d_path = "vec3d.exr"

    baseline_width = 0.065

    depth, disparity, skymap, detailmap_disp1 = get_depth(depth_path, intrinsics1, baseline_width)

    disparity2, flow2d, rigidmap, detailmap_disp2, detailmap_flow = get_sceneflow(vec3d_path, extrinsics1, extrinsics2, intrinsics1, intrinsics2, depth, disparity, skymap, baseline_width)

    flow_IO.writeDsp5File(disparity, "disparity.dsp5")
    flow_IO.writeDsp5File(disparity2, "disparity2.dsp5")
    flow_IO.writeFlo5File(flow2d, "flow.flo5")

    flow_IO.writePngMapFile(skymap, "skymap.png")
    flow_IO.writePngMapFile(detailmap_disp1, "detailmap_disp1.png")
    flow_IO.writePngMapFile(detailmap_disp2, "detailmap_disp2.png")
    flow_IO.writePngMapFile(detailmap_flow, "detailmap_flow.png")
    flow_IO.writePngMapFile(rigidmap, "rigidmap.png")

    # matchmaps are computed with matchmap(fw, bw)
    # where fw, bw are either optical flows or disparities after conversion with disp1ToFlow or disp2ToFlow
