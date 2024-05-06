import bpy
import os
import os.path
import blf
import bpy.types
import numpy as np

from .image_processing import Feature_detection, PlaneItem
from .blender_operations import DrawMesh, ResetImage

GlobalPlaneDataArray : list[PlaneItem] = [] # this will eventually replace the two array under this
PlaneAdded : bool = False
PlaneCreated : bool = False

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
        global PlaneAdded; PlaneAdded = True
        return {'FINISHED'}
  

class VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel(bpy.types.Panel):  
    bl_label = "Sketch input Section"
    bl_idname = "_PT_Views_File_Path"
    bl_parent_id = "_PT_Sketch_To_Mesh_Main_Panel" 
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    @classmethod
    def poll(self,context): return context.mode == 'OBJECT'
    
    def draw(self, context):
        layout = self.layout

        Firstbox = layout.box()
        row = Firstbox.row()
        Firstbox.prop(context.scene, "PlaneFilePath", text="Image file path") 
        row = Firstbox.row()
        row.prop(context.scene, "PlaneRotation", text="Rotation", slider=True) 
        row = Firstbox.row()
        Firstbox.operator("object.add_plane_item", text="Add Image")

        global PlaneAdded
        if PlaneAdded == True:
            Secondbox = layout.box()
            # List current plane items
            for item in GlobalPlaneDataArray:
                col = Secondbox.row()
                col.label(text="Name: " + os.path.basename(item.PlaneFilepath) + "Rotation: " + str(item.PlaneRotation))

            row = Secondbox.row()
            row.operator("object.place_image_in_space", text="Confirm Images")
            row = Secondbox.row()
            row.operator("object.reset_selected_images", text="Reset Images")


class PlaceImageIn3D(bpy.types.Operator):
    bl_idname = "object.place_image_in_space"
    bl_label = "Place Images"
    bl_description = "Sends images to feature detection" # rework possibly?

    def execute(self, context):
        Feature_detection(self=self, PlaneDataArray=GlobalPlaneDataArray)
        global PlaneCreated
        PlaneCreated = True
        return {'FINISHED'}


# this will need rework.
# TODO: figure out what of this is still usable later on
# - SaveMesh button will certainly be used later. It is currently doing nothing
class VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel(bpy.types.Panel):  
    bl_label = "MeshSettings"
    bl_idname = "_PT_MeshSettings"
    bl_parent_id = "_PT_Sketch_To_Mesh_Main_Panel"  # Set the parent panel ID
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    @classmethod
    def poll(self,context): return context.mode == 'OBJECT'

    def draw(self, context):
        global PlaneCreated
        if PlaneCreated == True:
            layout = self.layout
            layout.label(text="Mesh Settings")
            row = layout.row()
            row.prop(context.scene, "poly_count_range", text="Vertices Separator", slider=True)
            row = layout.row()
            layout.label(text="Save Mesh")
            row = layout.row()
            layout.prop(context.scene, "IsComplex", text="Is Complex")
            row = layout.row()
            layout.prop(context.scene, "FileName_Input", text="")
            row = layout.row()
            row.operator("wm.place_mesh", text="Place Mesh")
            row = layout.row()
            row.operator("object.reset_mesh_collection", text="Reset Mesh")
            row = layout.row()
            row.operator("wm.database_export", text="Export File")
            row = layout.row()
            row.prop(context.scene, "mesh_rating", text="Mesh Rating", slider=True)


class Reset_Input_Images(bpy.types.Operator): 
    bl_idname = "object.reset_selected_images"
    bl_label = "Reset_Images"
    bl_description = "Reset previously selected images"

    def execute(self, context):
        ResetImage(GlobalPlaneDataArray)
        global PlaneCreated; PlaneCreated = False
        global PlaneAdded; PlaneAdded = False
        return {'FINISHED'}
    

class Reset_Mesh_Collection(bpy.types.Operator): 
    bl_idname = "object.reset_mesh_collection"
    bl_label = "Reset_Mesh"
    bl_description = "Resets the mesh previously created also resets the input images"

    def execute(self, context):
        ResetImage(GlobalPlaneDataArray)
        global PlaneCreated; PlaneCreated = False
        global PlaneAdded; PlaneAdded = False
        collection = bpy.data.collections.get("Sketch_to_Mesh_Collection")
        if collection is not None: bpy.data.collections.remove(collection)
        return {'FINISHED'}

class PlaceMesh(bpy.types.Operator):
    bl_idname = "wm.place_mesh"
    bl_label ="Place Mesh"

    def execute(self, context):
        DrawMesh((0, 255, 0), bpy.context.scene.poly_count_range, self, GlobalPlaneDataArray, bpy.context.scene.IsComplex)
        return {'FINISHED'}
