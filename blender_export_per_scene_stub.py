import sys
import os
import bpy

import_dir = bpy.path.abspath('//../../..')

if import_dir not in sys.path:
   sys.path.append(import_dir)

import importlib

blender_export = importlib.import_module('blender_export')
importlib.reload(blender_export)


VIEW_CELL_CENTER = [ -18.0, -45.0, 5.0]
VIEW_CELL_SIZE = [2.0, 2.0, 2.0]
VIEW_ROT_START = [90.0,0,0]
VIEW_ROT_RESTR = [20,0,30]
VIEWS = 100
SCENE_NAME = 'forest_4k'
RESOLUTION_X = 3840
RESOLUTION_Y = 2160

CAM_NAME = "renderCam"

blender_export.export_view_cells(VIEWS=VIEWS, VIEW_CELL_CENTER=VIEW_CELL_CENTER, VIEW_CELL_SIZE=VIEW_CELL_SIZE,
 VIEW_ROT_START=VIEW_ROT_START, VIEW_ROT_RESTR=VIEW_ROT_RESTR, SCENE_NAME=SCENE_NAME, CAM_NAME=CAM_NAME, RESOLUTION_X=RESOLUTION_X, RESOLUTION_Y=RESOLUTION_Y)

