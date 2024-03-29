import bpy
import os
import os.path
import blf
import bpy.types
from os import path
import numpy as np
from dataclasses import dataclass
from .image_processing import Feature_detection, PlaneItem
from .blender_operations import DrawAllMeshesToScreen


@dataclass
class UserData:
    UserSignedIn = False
    user_info = []
    user_documents = []
    def __init__(self, SignIn):
        self.user_info = []
        self.user_documents = [] # testing
        self.UserSignedIn = SignIn
User: UserData = UserData(False)
GlobalPlaneDataArray : list[PlaneItem] = [] # this will eventually replace the two array under this

def draw_callback_px(self, context, message):
    font_id = 0
    blf.position(font_id, 15, 30, 0)
    blf.size(font_id, 20)
    blf.draw(font_id, message)
    
class NotificationPopup(bpy.types.Operator):
    bl_idname = "wm.toast_notification"
    bl_label = "Show Toast Notification"
    
    message: bpy.props.StringProperty(name="Message",description="The message to display in the toast",default="Toast Notification!" )

    def execute(self, context):
        args = (self, context, self.message)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
        self.report({'INFO'}, "OK Pressed")
        return {'FINISHED'}

    def invoke(self, context, event):    
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.label(text=self.message)
  
# Operator to add a new plane item
# adds new image to be analyzed
class OBJECT_OT_add_plane_item(bpy.types.Operator):
    bl_idname = "object.add_plane_item"
    bl_label = "Add Plane Item"
    bl_description = "Select and add new images to be processed"

    def execute(self, context):
        #adds the plane Itme to the Plane Item List
        NewFileRotationPair = PlaneItem(bpy.context.scene.PlaneFilePath, bpy.context.scene.PlaneRotation )
        GlobalPlaneDataArray.append(NewFileRotationPair)
        return {'FINISHED'}

    def draw(self, context):    
        layout = self.layout   
        row = layout.row()
        row.prop(context.scene, "PlaneFilePath", text="FilePath ") 
        row = layout.row()
        row.prop(context.scene, "PlaneRotation", text="Rotation ", slider=True) 

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel(bpy.types.Panel):  
    bl_label = "Views"
    bl_idname = "_PT_Views_File_Path"
    bl_parent_id = "_PT_Sketch_To_Mesh_Main_Panel" 
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    @classmethod
    def poll(self,context):
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


class PlaceImageIn3D(bpy.types.Operator):
    bl_idname = "object.place_image_in_space"
    bl_label = "Place Images"
    bl_description = "Sends images to feature detection" # rework possibly?

    def execute(self, context):
        Feature_detection(self=self, PlaneDataArray=GlobalPlaneDataArray)
        return {'FINISHED'}


class Reset_Input_Images(bpy.types.Operator): 
    bl_idname = "object.reset_selected_images"
    bl_label = "Reset_Images"
    bl_description = "Reset previously selected images"

    def execute(self, context):
        Itervalue = 0

        for plane_data in GlobalPlaneDataArray :
            #Finds the Images Saved
            ImageFilePath = os.path.abspath("ImageFolder\\" + "View" + str(Itervalue) + plane_data.PlaneFilepath[plane_data.PlaneFilepath.rfind("."): ] ) 
            if path.exists(ImageFilePath): 
                os.remove(ImageFilePath) # if we find that file we will delete it
                # selects the image plane in the array 
                bpy.data.objects[plane_data.ImagePlaneName].select_set(True)
                #deletes the image plane in the array
                bpy.ops.object.delete(use_global=False, confirm=False)
                #increases the itervale to reach the next View string
            Itervalue = Itervalue + 1 

        # clears the PlaneData Array
        GlobalPlaneDataArray.clear() 
        return {'FINISHED'}


class TestPlaceMesh(bpy.types.Operator):
    bl_idname = "wm.place_mesh"
    bl_label ="Place Mesh"

    def execute(self, context):
        DrawAllMeshesToScreen((0, 255, 0), 5, self, GlobalPlaneDataArray)
        return {'FINISHED'}
