from typing import Set
import bpy
import io
from pymongo import MongoClient
from bpy.types import Context
from bpy_extras.io_utils import ExportHelper

class PingDB(bpy.types.Operator):
    bl_idname = "object.do_ping"
    bl_label = "Ping Db"

    def execute(self, context):
        mClient = MongoClient('mongodb+srv://devdb:dev123@cluster0.aukmt1u.mongodb.net/?retryWrites=true&w=majority')
        db = mClient['StM-dev']
        collection = db['Test']

        blend_file_path = r"C:\Users\James Burns\Documents\untitled.blend"

        with open(blend_file_path, "rb") as file:
            blend_file_contents = io.BytesIO(file.read())

        data = {"filename": "untitled.blend", "data": blend_file_contents.getvalue()}

        collection.insert_one(data)

        mClient.close()
        return {'FINISHED'}
    
class ErrorHandling(bpy.types.Operator):
    bl_idname = "object.error"
    bl_label = "Error!!"

    errorMessage = "Please select an image file"
    myFilePath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        if not self.myFilePath.endswith(".png"):
            context.window_manager.invoke_popup(self)
        return {'FINISHED'}

class DoImg(bpy.types.Operator):
    bl_idname = "object.do_img"
    bl_label = "Place Image"
    
    #Property that holds the filepath that will be used to insert the image
    myFilePath: bpy.props.StringProperty(subtype="FILE_PATH")

    #This is the function that inserts the image into blender
    def execute(self, context):
        bpy.ops.object.load_reference_image(filepath=self.myFilePath)
        return {'FINISHED'}
    
    #This is a function that opens a file explorer
    #THIS DOES NOT DO ANYTHING I just think its going to be useful to have later
    def invoke(self, context, event):
        # Open a file browser to select a file
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class LayoutDemoPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Layout Demo"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout

        # Big save button
        layout.label(text="Big Save Button:")
        row = layout.row()
        row.scale_y = 2.0
        row.operator("wm.save_mainfile")
        
        #front file path
        row = layout.row()
        row.prop(context.scene, "front_views_file_path", text="Front View")
        do_error_checking = row.operator("object.error")
        do_error_checking.myFilePath = context.scene.front_views_file_path

        #right file path
        row = layout.row()
        row.prop(context.scene, "right_views_file_path", text="Right View")

        #left file path
        row = layout.row()
        row.prop(context.scene, "left_views_file_path", text="Left View")

        #back file path
        row = layout.row()
        row.prop(context.scene, "back_views_file_path", text="Back View")

        #top file path
        row = layout.row()
        row.prop(context.scene, "top_views_file_path", text="Top View")

        #bottom file path
        row = layout.row()
        row.prop(context.scene, "bottom_views_file_path", text="Bottom View")
        
        # Big Ping button
        layout.label(text="Big Ping Button:")
        row = layout.row()
        row.scale_y = 2.0
        do_db_ping = row.operator("object.do_ping")
        
        # Big image button
        layout.label(text="Big Import Button:")
        row = layout.row()
        row.scale_y = 2.0
        #The operator is a DoImg object
        do_img_op = row.operator("object.do_img")
        #The filepath for the image is passed in here to be used
        do_img_op.myFilePath = context.scene.front_views_file_path


#A function that initializes all of the classes and views in the file
def register():
    bpy.utils.register_class(LayoutDemoPanel)
    bpy.utils.register_class(DoImg)
    bpy.utils.register_class(ErrorHandling)
    bpy.utils.register_class(PingDB)
    bpy.types.Scene.front_views_file_path = bpy.props.StringProperty(subtype="FILE_PATH")
    bpy.types.Scene.right_views_file_path = bpy.props.StringProperty(subtype="FILE_PATH")
    bpy.types.Scene.left_views_file_path = bpy.props.StringProperty(subtype="FILE_PATH")
    bpy.types.Scene.back_views_file_path = bpy.props.StringProperty(subtype="FILE_PATH")
    bpy.types.Scene.top_views_file_path = bpy.props.StringProperty(subtype="FILE_PATH")
    bpy.types.Scene.bottom_views_file_path = bpy.props.StringProperty(subtype="FILE_PATH")

#A functoin that deconstructs the classes and views in the file
def unregister():
    bpy.utils.unregister_class(LayoutDemoPanel)
    bpy.utils.unregister_class(DoImg)
    bpy.utils.unregister_class(ErrorHandling)
    bpy.utils.unregister_class(PingDB)
    bpy.types.Scene.front_views_file_path
    bpy.types.Scene.right_views_file_path
    bpy.types.Scene.left_views_file_path
    bpy.types.Scene.back_views_file_path
    bpy.types.Scene.top_views_file_path
    bpy.types.Scene.bottom_views_file_path 


#This calls register
if __name__ == "__main__":
    register()