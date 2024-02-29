bl_info = {
    "name": "Sketch_To_Mesh",
    "author": "LuckyNinjas",
    "version": (0, 0, 1),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > My Custom Panel category",
    "description": "The Inital UI skeleton",
    "category": "Development",
}
    
import bpy

from .ui_operations import OBJECT_OT_add_plane_item,  Reset_Input_Images, VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel, PlaceImageIn3D, DataBaseLogin, DataBaseRegister, DataBaseUIMenu, UserAccessDb, DocumentItem, DataBaseLogout, NotificationPopup
from .testing_operations import DoImg, StMTestImagePrep, StMTestSaveFileToDb, StMTestConnectionOperator, StMTestGetFileFromDbFromUserId, StMTestDeleteFileFromDbFromUserId
from .base_ui import VIEW3D_PT_Sketch_To_Mesh_Panel, VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel, VIEW3D_PT_Sketch_To_Mesh_Testing, AccessDbCustomPanel

def register():
    bpy.types.Scene.poly_count_range = bpy.props.IntProperty(name="Poly Count", default=10, min=0, max=100)
    bpy.types.Scene.mesh_rating = bpy.props.IntProperty(name="Mesh Rating", default=10, min=0, max=100)
    bpy.types.Scene.Image_Center_X = bpy.props.IntProperty(name="Image Center X", default=10, min=0, max=100)
    bpy.types.Scene.Image_Center_Y = bpy.props.IntProperty(name="Image Center Y", default=10, min=0, max=100)
    bpy.types.Scene.FileName_Input = bpy.props.StringProperty(name="FileName", default="STMFile")
    #Database Properties
    bpy.types.Scene.DB_Username = bpy.props.StringProperty(name="DBUsername", default="")
    bpy.types.Scene.DB_Password = bpy.props.StringProperty(name="DBPassword", default="")
    #Plane data Properites
    bpy.types.Scene.PlaneFilePath = bpy.props.StringProperty(name="File Path",subtype='FILE_PATH')
    bpy.types.Scene.PlaneRotation = bpy.props.IntProperty(name="Image Center Y", default=0, min=-180, max=180)
    #Classes
    bpy.utils.register_class(OBJECT_OT_add_plane_item)
    bpy.utils.register_class(Reset_Input_Images)
    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_Panel)
    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel)
    bpy.utils.register_class(PlaceImageIn3D)
    bpy.utils.register_class(DoImg)
    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel)
    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_Testing)
    bpy.utils.register_class(NotificationPopup)

    # db
    bpy.utils.register_class(UserAccessDb)
    bpy.utils.register_class(AccessDbCustomPanel) 
    bpy.utils.register_class(DataBaseLogout) 
    
    bpy.utils.register_class(DocumentItem)
    bpy.types.Scene.my_document_collection = bpy.props.CollectionProperty(type=DocumentItem)
    bpy.types.Scene.my_document_index = bpy.props.IntProperty()
    
    # Tests
    bpy.utils.register_class(DataBaseUIMenu)
    bpy.utils.register_class(DataBaseRegister)
    bpy.utils.register_class(DataBaseLogin)
    bpy.utils.register_class(StMTestImagePrep)  
    bpy.utils.register_class(StMTestSaveFileToDb) 
    bpy.utils.register_class(StMTestConnectionOperator) 
    bpy.utils.register_class(StMTestGetFileFromDbFromUserId) 
    bpy.utils.register_class(StMTestDeleteFileFromDbFromUserId) 
    

def unregister():
    del bpy.types.Scene.poly_count_range
    del bpy.types.Scene.mesh_rating
    del bpy.types.Scene.Image_Center_X
    del bpy.types.Scene.Image_Center_Y
    del bpy.types.Scene.FileName_Input

    del bpy.types.Scene.DB_Username
    del bpy.types.Scene.DB_Password

    del bpy.types.Scene.PlaneFilePath
    del bpy.types.Scene.PlaneRotation
    #Classes
    bpy.utils.unregister_class(OBJECT_OT_add_plane_item)
    bpy.utils.unregister_class(Reset_Input_Images)
    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_Panel)
    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel)
    bpy.utils.unregister_class(PlaceImageIn3D)
    bpy.utils.unregister_class(DoImg)
    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel)
    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_Testing)
    bpy.utils.unregister_class(NotificationPopup)

    # db
    bpy.utils.unregister_class(UserAccessDb)
    bpy.utils.unregister_class(AccessDbCustomPanel)
    bpy.utils.unregister_class(DataBaseLogout)
    bpy.utils.unregister_class(DocumentItem)
    del bpy.types.Scene.my_document_collection
    del bpy.types.Scene.my_document_index

    # db test connection and image prep
    # Tests
    bpy.utils.unregister_class(DataBaseUIMenu)
    bpy.utils.unregister_class(DataBaseRegister)
    bpy.utils.unregister_class(DataBaseLogin)
    bpy.utils.unregister_class(StMTestImagePrep)
    bpy.utils.unregister_class(StMTestConnectionOperator)
    bpy.utils.unregister_class(StMTestSaveFileToDb)
    bpy.utils.unregister_class(StMTestGetFileFromDbFromUserId)
    bpy.utils.unregister_class(StMTestDeleteFileFromDbFromUserId) 

if __name__ == "__main__":
    register()