import bpy
from .DatabaseUI import User
# this contains the main layout for the Sketch to mesh program
# to link up functions with the buttons
# first create the operator 
# find the panel you want the funciton in(this cannot be inside the main function since everything is not inside the main panel except for the other panels)
# use ( row.operator("(the operator you want)", text="(the name of the button)"))
# for now all of the button will create a cube

class VIEW3D_PT_Sketch_To_Mesh_Panel(bpy.types.Panel):  
    bl_idname = "_PT_Sketch_To_Mesh_Main_Panel" 
    bl_label = "Sketch-To-Mesh"  # found at the top of the Panel
    bl_space_type = "VIEW_3D"  
    bl_region_type = "UI"  
    bl_category = "S-T-M"  # Sidebar cName
    
    def draw(self, context):  layout = self.layout

# this will need rework.
# TODO: figure out what of this is still usable later on
# - SaveMesh button will certainly be used later. It is currently doing nothing
class VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel(bpy.types.Panel):  
    bl_label = "MeshSettings"
    bl_idname = "_PT_MeshSettings"
    bl_parent_id = "_PT_Sketch_To_Mesh_Main_Panel"  # Set the parent panel ID
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
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

class DataBaseUIMenu(bpy.types.Panel):
    bl_idname = "wm.database_ui_menu"
    bl_label = "Database Menu"
    bl_parent_id = "_PT_Sketch_To_Mesh_Main_Panel" 
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):    
        layout = self.layout
        row = layout.row()
        if User.UserSignedIn == False :
            row.operator("wm.database_register", text="Register User")
            row = layout.row()
            row.operator("wm.database_login", text="Login User")
        else :
            row.operator("wm.access_database", text="Access Database") 
            row = layout.row() 
            row.operator("wm.user_logout", text="Logout") # TODO: logout function in authentication

class VIEW3D_PT_Sketch_To_Mesh_Testing(bpy.types.Panel):  
    bl_label = "Testing"
    bl_idname = "_PT_Testing_Panel"
    bl_parent_id = "_PT_Sketch_To_Mesh_Main_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("wm.test_connection_operator", text="Test Connection")
        row = layout.row()
        row.operator("wm.prepare_image_operator", text="Test Image Prep")
        row = layout.row()
        row.operator("wm.save_file_to_db_operator", text="Save File to DB")
        row = layout.row()
        row.operator("wm.get_file_from_db_operator", text="Get File from DB")
        row = layout.row()
        row.operator("wm.delete_file_from_db_operator", text="Delete File from DB")
        row = layout.row()
        row.operator("wm.toast_notification", text="Toast Test")

