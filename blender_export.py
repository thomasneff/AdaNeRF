import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
import random as rnd
from math import radians
from mathutils import Vector

from types import SimpleNamespace

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def getRndCameraPos(huge_dict):
    n1 = rnd.random()
    n2 = rnd.random()
    n3 = rnd.random()
    return ( huge_dict.VIEW_CELL_CENTER[0] + (n1 - 0.5) * huge_dict.VIEW_CELL_SIZE[0], huge_dict.VIEW_CELL_CENTER[1] + (n2 - 0.5) * huge_dict.VIEW_CELL_SIZE[1],huge_dict.VIEW_CELL_CENTER[2] + (n3 - 0.5) * huge_dict.VIEW_CELL_SIZE[2])

def getRndCameraRot(huge_dict):
    n1 = huge_dict.VIEW_ROT_START[0] +( ( rnd.random() - 0.5) * huge_dict.VIEW_ROT_RESTR[0] )
    n2 = huge_dict.VIEW_ROT_START[1] +( ( rnd.random() - 0.5) * huge_dict.VIEW_ROT_RESTR[1] )
    n3 = huge_dict.VIEW_ROT_START[2] +( ( rnd.random() - 0.5) * huge_dict.VIEW_ROT_RESTR[2] )

    return radians(n1) ,radians(n2) ,radians(n3)


def renderSet(huge_dict, fp, scene, subf, num_views):
    out_data = {
        'camera_angle_x': bpy.data.objects[huge_dict.CAM_NAME].data.angle_x,
        'view_cell_center': huge_dict.VIEW_CELL_CENTER,
        'view_cell_size': huge_dict.VIEW_CELL_SIZE,
        'random_seed': huge_dict.SEED,
    }

    print(f"VIEW CELL CENTER {out_data['view_cell_center']}")

    cam = scene.objects[huge_dict.CAM_NAME]

    cam.rotation_euler[0] =radians( huge_dict.VIEW_ROT_START[0] )
    cam.rotation_euler[1] =radians( huge_dict.VIEW_ROT_START[1] )
    cam.rotation_euler[2] =radians( huge_dict.VIEW_ROT_START[2] )
    cam.location = (huge_dict.VIEW_CELL_CENTER[0], huge_dict.VIEW_CELL_CENTER[1], huge_dict.VIEW_CELL_CENTER[2])
    bpy.context.view_layer.update()
    print(listify_matrix(cam.matrix_world))
    out_data['camera_base_orientation'] = listify_matrix(cam.matrix_world)

    out_data['frames'] = []


    for i in range(huge_dict.VIEWS_OFFSET, huge_dict.VIEWS_OFFSET+num_views):
        print(f"(Rendering {subf} file {i}")
        scene.render.filepath = fp + f'/{subf}/' + f'{i:05d}' + '.png'

        cam.location = getRndCameraPos(huge_dict)

        r1, r2, r3 = getRndCameraRot(huge_dict)

        cam.rotation_euler[0] = r1
        cam.rotation_euler[1] = r2
        cam.rotation_euler[2] = r3

        skip_existing = False

        if os.path.exists(fp + f'/{subf}/' + f'{i:05d}' + '_depth.npz') and huge_dict.SKIP_EXISTING_FILES:
            print(f"Skipping existing file {scene.render.filepath}!")
            skip_existing = True

        if not huge_dict.DEBUG and not skip_existing:
            bpy.ops.render.render(write_still=True)

            pixels = bpy.data.images[huge_dict.RENDER_IMG_NAME].pixels

            # copy buffer to numpy array for faster manipulation
            arr = np.array(pixels[:])
            depth = np.array(arr.reshape(-1,4)[:,0], dtype=np.float32)
            print(depth.max())

            fcp = fp + f'/{subf}/' + f'{i:05d}' + '_depth.npz'
            np.savez(fcp, depth)

        bpy.context.view_layer.update()
        frame_data = {
            'file_path': f"./{subf}/{i:05d}",
            'rotation': 0,
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)
        with open(fp + '/' + f'transforms_{subf}_save.json', 'a')as out_file:
            json.dump(frame_data, out_file, indent=4)

    if not huge_dict.DEBUG:
        #print(out_data)
        with open(fp + '/' + f'transforms_{subf}.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
    else:
        print(out_data)


def create_set(huge_dict):
    fp = bpy.path.abspath(f"//{huge_dict.RESULTS_PATH}")

    ensure_dir(fp)

    scene = bpy.context.scene
    scene.render.resolution_x = huge_dict.RESOLUTION_X
    scene.render.resolution_y = huge_dict.RESOLUTION_Y
    scene.render.resolution_percentage = 100
    cam = scene.objects[huge_dict.CAM_NAME]
    scene.render.image_settings.file_format = 'PNG'

    renderSet(huge_dict, fp,scene,'train', huge_dict.VIEWS_TRAIN)
    renderSet(huge_dict, fp,scene,'test', huge_dict.VIEWS_TEST)
    renderSet(huge_dict, fp,scene,'val', huge_dict.VIEWS_VAL)

def export_view_cells(SEED=42, DEBUG=False, VIEWS_CAM_PATH=40, VIEWS=None, VIEWS_TRAIN=None, VIEWS_VAL=None, VIEWS_TEST=None, VIEWS_OFFSET=0, RESOLUTION=800, RESOLUTION_X=None, RESOLUTION_Y=None, COLOR_DEPTH=8, FORMAT='PNG', VIEW_CELL_CENTER=None, VIEW_CELL_SIZE=None, VIEW_ROT_START=None, VIEW_ROT_RESTR=None, SKIP_EXISTING_FILES=True, CAM_NAME="renderCam", RENDER_IMG_NAME="Viewer Node", SCENE_NAME=None):



    if SCENE_NAME is None:
        print("Error: Please specify SCENE_NAME!")
        return

    huge_dict = SimpleNamespace()
    huge_dict.SEED = SEED

    rnd.seed(huge_dict.SEED)

    huge_dict.DEBUG = DEBUG
    huge_dict.VIEWS_CAM_PATH = VIEWS_CAM_PATH
    huge_dict.VIEWS = VIEWS
    huge_dict.VIEWS_TRAIN = VIEWS_TRAIN
    huge_dict.VIEWS_VAL = VIEWS_VAL
    huge_dict.VIEWS_TEST = VIEWS_TEST

    if huge_dict.VIEWS_TRAIN is None:
        huge_dict.VIEWS_TRAIN = huge_dict.VIEWS

    if huge_dict.VIEWS_VAL is None:
        huge_dict.VIEWS_VAL = huge_dict.VIEWS

    if huge_dict.VIEWS_TEST is None:
        huge_dict.VIEWS_TEST = huge_dict.VIEWS

    if VIEW_CELL_CENTER is None or VIEW_CELL_SIZE is None or VIEW_ROT_START is None or VIEW_ROT_RESTR is None or huge_dict.VIEWS_TRAIN is None or huge_dict.VIEWS_VAL is None or huge_dict.VIEWS_TEST is None:
        print("Error: VIEW_CELL* and VIEWS_* parameters need to be specified!")
        return


    huge_dict.VIEWS_OFFSET = VIEWS_OFFSET
    huge_dict.RESOLUTION = RESOLUTION

    huge_dict.RESOLUTION_X = RESOLUTION
    huge_dict.RESOLUTION_Y = RESOLUTION

    if RESOLUTION_X is not None and RESOLUTION_Y is not None:
        huge_dict.RESOLUTION_X = RESOLUTION_X
        huge_dict.RESOLUTION_Y = RESOLUTION_Y


    huge_dict.COLOR_DEPTH = COLOR_DEPTH
    huge_dict.FORMAT = FORMAT
    huge_dict.VIEW_CELL_CENTER = VIEW_CELL_CENTER
    huge_dict.VIEW_CELL_SIZE = VIEW_CELL_SIZE
    huge_dict.VIEW_ROT_START = VIEW_ROT_START
    huge_dict.VIEW_ROT_RESTR = VIEW_ROT_RESTR
    huge_dict.SKIP_EXISTING_FILES = SKIP_EXISTING_FILES
    huge_dict.CAM_NAME = CAM_NAME
    huge_dict.RENDER_IMG_NAME = RENDER_IMG_NAME

    huge_dict.RESULTS_PATH = SCENE_NAME + '_' + str(VIEW_CELL_CENTER) + '_' + str(VIEW_CELL_SIZE) + '_' + str(VIEW_ROT_RESTR) + '_' + str(VIEWS)

    create_set(huge_dict)

    # Reset to initial cam position
    cam = bpy.context.scene.objects[huge_dict.CAM_NAME]

    cam.rotation_euler[0] =radians( huge_dict.VIEW_ROT_START[0] )
    cam.rotation_euler[1] =radians( huge_dict.VIEW_ROT_START[1] )
    cam.rotation_euler[2] =radians( huge_dict.VIEW_ROT_START[2] )
    cam.location = (huge_dict.VIEW_CELL_CENTER[0], huge_dict.VIEW_CELL_CENTER[1], huge_dict.VIEW_CELL_CENTER[2])
    bpy.context.view_layer.update()

