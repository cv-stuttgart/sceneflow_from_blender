import bpy
import numpy as np
from mathutils import Matrix


def get_intrinsics(focal_length, sizex=1920, sizey=1080, sensor_width=23.76):
    # only works for sensor fit "Horizontal"
    f = focal_length * sizex / sensor_width
    return [f, f, sizex//2, sizey//2]


def get_3x4_RT_matrix_from_blender(cam):
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    T_world2bcam = -1*R_world2bcam @ location

    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT


# TODO: select main camera here
cam = bpy.data.objects["Camera_L"]

worldmat = get_3x4_RT_matrix_from_blender(cam)
worldmat = np.vstack((worldmat, np.asarray([[0,0,0,1]])))
np.savetxt(f"extrinsics.txt", worldmat)

intrinsics = get_intrinsics(cam.data.lens)
np.savetxt(f"intrinsics.txt", intrinsics)
