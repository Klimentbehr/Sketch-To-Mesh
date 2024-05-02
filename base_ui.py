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
    bl_category = "S-T-M"  # Sidebar Name
    
    def draw(self, context):  layout = self.layout

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