import bpy
import os
import cv2
import os.path
from os import path
from dataclasses import dataclass
from bpy.props import StringProperty, IntProperty, BoolProperty, CollectionProperty
from bpy.types import Operator, Panel
#from image_processing import prepare_image, detect_and_describe_akaze, outline_image, match_features, draw_matches
#from bcrypt_password import hash_password
#from authentication import login_account, register_account

# Data class for plane item
@dataclass
class PlaneItem:
    PlaneFilepath: bpy.props.StringProperty(name="File Path", subtype='FILE_PATH')
    PlaneRotation: bpy.props.IntProperty(name="Rotation", default=0)
    ImagePlaneName: str = ""
    ImagePlaneFilePath: str = ""
    isSymmetrical: bool = False

    def __init__(self, filepath, rotation, isSymmetrical=False):
        self.PlaneFilepath = filepath
        self.PlaneRotation = rotation
        self.isSymmetrical = isSymmetrical

# Data class for user data
@dataclass
class UserData:
    UserSignedIn: bool = False

    def __init__(self, SignIn):
        self.UserSignedIn = SignIn

User: UserData = UserData(False)
GlobalPlaneDataArray: list[PlaneItem] = []  # this will eventually replace the two array under this
isSymmetrical = False  # Bool to switch between symmetrical and default function paths

# Operator to add a new plane item
class OBJECT_OT_add_plane_item(bpy.types.Operator):
    bl_idname = "object.add_plane_item"
    bl_label = "Add Plane Item"
    isSymmetrical: BoolProperty(name="Is Symmetrical", default=False)
    isComplex: BoolProperty(name="Is Complex", default=False)  # Add the boolean property

    def execute(self, context):
        # Adds the plane item to the Plane Item List
        NewFileRotationPair = PlaneItem(bpy.context.scene.PlaneFilePath, bpy.context.scene.PlaneRotation, self.isSymmetrical)
        GlobalPlaneDataArray.append(NewFileRotationPair)
        return {'FINISHED'}

    def draw(self, context):    
        layout = self.layout   
        row = layout.row()
        row.prop(context.scene, "PlaneFilePath", text="File Path")
        row = layout.row()
        row.prop(context.scene, "PlaneRotation", text="Rotation", slider=True)
        row = layout.row()
        row.prop(self, "isSymmetrical", text="Is Symmetrical")
        row = layout.row()
        row.prop(self, "isComplex", text="Is Complex")  # Add the property to the UI

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

# Panel for displaying plane item views
class VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel(bpy.types.Panel):  
    bl_label = "Views"
    bl_idname = "_PT_Views_File_Path"
    bl_parent_id = "_PT_Sketch_To_Mesh_Main_Panel" 
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    @classmethod
    def poll(self, context):
        return context.mode == 'OBJECT'
    
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        layout.operator("object.add_plane_item", text="Add Image")

        box = layout.box()

        # List current plane items
        for item in GlobalPlaneDataArray:
            col = box.row()
            col.label(text="Name: " + os.path.basename(item.PlaneFilepath) + "Rotation: " + str(item.PlaneRotation))

        row = layout.row()
        row.operator("object.place_image_in_space", text="Confirm Images")
        row = layout.row()
        row.operator("object.reset_selected_images", text="Reset Images")

# Operator for placing images in 3D space
class PlaceImageIn3D(bpy.types.Operator):
    bl_idname = "object.place_image_in_space"
    bl_label = "Place Images"

    def execute(self, context):
        # Your feature detection function call here
        # Example: Feature_detection(self=self, PlaneDataArray=GlobalPlaneDataArray)
        return {'FINISHED'}

# Operator for resetting selected images
class Reset_Input_Images(bpy.types.Operator): 
    bl_idname = "object.reset_selected_images"
    bl_label = "Reset Images"

    def execute(self, context):
        # Your reset image function implementation here
        return {'FINISHED'}

def register():
    bpy.utils.register_class(OBJECT_OT_add_plane_item)
    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_add_plane_item)
    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel)

if __name__ == "__main__":
    register()
