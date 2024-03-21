import bpy

def blend_opener(file_path):
    # open up the blend file here
    bpy.ops.wm.open_mainfile(filepath=file_path)
    return 0

def fbx_opener(file_path):
    # open up the .fbx file here
    bpy.ops.import_scene.fbx(filepath=file_path)
    return 0