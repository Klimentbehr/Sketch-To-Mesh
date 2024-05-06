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

from .ui_operations import OBJECT_OT_add_plane_item, Reset_Input_Images, VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel, PlaceImageIn3D, PlaceMesh,  NotificationPopup, VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel, Reset_Mesh_Collection
from .testing_operations import DoImg,  ExportToDatabase#, StMTestImagePrep, StMTestSaveFileToDb, StMTestConnectionOperator, StMTestGetFileFromDbFromUserId, StMTestDeleteFileFromDbFromUserId,
from .DatabaseUI import DataBaseLogin, DataBaseRegister, DocumentItem, DataBaseLogout,  DataBase_UIList, DeleteFromDatabase, AccessDatabase, AddToDatabase, ImportFromDataBase
from .base_ui import VIEW3D_PT_Sketch_To_Mesh_Panel, DataBaseUIMenu#, VIEW3D_PT_Sketch_To_Mesh_Testing

def register():
  
    bpy.types.Scene.poly_count_range = bpy.props.IntProperty(name="Vertice Separation Modifer", default=10, min=0, max=100)

    bpy.types.Scene.mesh_rating = bpy.props.IntProperty(name="Mesh Rating", default=10, min=0, max=100)
    bpy.types.Scene.Image_Center_X = bpy.props.IntProperty(name="Image Center X", default=10, min=0, max=100)
    bpy.types.Scene.Image_Center_Y = bpy.props.IntProperty(name="Image Center Y", default=10, min=0, max=100)
    bpy.types.Scene.FileName_Input = bpy.props.StringProperty(name="FileName", default="STMFile")
    bpy.types.Scene.IsComplex = bpy.props.BoolProperty(name="isComplex", default=False)

    #Database Properties
    bpy.types.Scene.DB_Username = bpy.props.StringProperty(name="DBUsername", default="")
    bpy.types.Scene.DB_Password = bpy.props.StringProperty(name="DBPassword", default="")

    #Plane data Properites
    bpy.types.Scene.PlaneFilePath = bpy.props.StringProperty(name="File Path",subtype='FILE_PATH')
    bpy.types.Scene.PlaneRotation = bpy.props.IntProperty(name="Image Center Y", default=0, min=-180, max=180)

    #Classes
    bpy.utils.register_class(OBJECT_OT_add_plane_item)
    bpy.utils.register_class(Reset_Input_Images)
    bpy.utils.register_class(Reset_Mesh_Collection)

    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_Panel)
    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel)
    bpy.utils.register_class(PlaceImageIn3D)
    bpy.utils.register_class(DoImg)
    bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel)
    #bpy.utils.register_class(VIEW3D_PT_Sketch_To_Mesh_Testing)
    bpy.utils.register_class(NotificationPopup)

    # db
    bpy.utils.register_class(DocumentItem)    
    bpy.types.Scene.my_document_collection = bpy.props.CollectionProperty(type=DocumentItem)
    bpy.types.Scene.my_document_index = bpy.props.IntProperty()
    bpy.utils.register_class(AddToDatabase)
    #bpy.utils.register_class(AccessDbCustomPanel) 
    bpy.utils.register_class(AccessDatabase)
    bpy.utils.register_class(DataBaseLogout) 
    bpy.utils.register_class(DataBase_UIList)
    bpy.utils.register_class(DeleteFromDatabase)
    bpy.utils.register_class(ImportFromDataBase)
    
    # Tests
    bpy.utils.register_class(PlaceMesh)
    bpy.utils.register_class(ExportToDatabase)
    bpy.utils.register_class(DataBaseUIMenu)
    bpy.utils.register_class(DataBaseRegister)
    bpy.utils.register_class(DataBaseLogin)
    #bpy.utils.register_class(StMTestImagePrep)  
    #bpy.utils.register_class(StMTestSaveFileToDb) 
    #bpy.utils.register_class(StMTestConnectionOperator) 
    #bpy.utils.register_class(StMTestGetFileFromDbFromUserId) 
    #bpy.utils.register_class(StMTestDeleteFileFromDbFromUserId) 
    

def unregister():
    del bpy.types.Scene.poly_count_range
    del bpy.types.Scene.mesh_rating
    del bpy.types.Scene.Image_Center_X
    del bpy.types.Scene.Image_Center_Y
    del bpy.types.Scene.FileName_Input
    del bpy.types.Scene.IsComplex

    del bpy.types.Scene.DB_Username
    del bpy.types.Scene.DB_Password

    del bpy.types.Scene.PlaneFilePath
    del bpy.types.Scene.PlaneRotation
    #Classes
    bpy.utils.unregister_class(OBJECT_OT_add_plane_item)
    bpy.utils.unregister_class(Reset_Input_Images)
    bpy.utils.unregister_class(Reset_Mesh_Collection)

    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_Panel)
    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel)
    bpy.utils.unregister_class(PlaceImageIn3D)
    bpy.utils.unregister_class(DoImg)
    bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel)
    #bpy.utils.unregister_class(VIEW3D_PT_Sketch_To_Mesh_Testing)
    bpy.utils.unregister_class(NotificationPopup)

    # db
    bpy.utils.unregister_class(AddToDatabase)
    #bpy.utils.unregister_class(AccessDbCustomPanel)
    bpy.utils.unregister_class(DataBaseLogout)
    bpy.utils.unregister_class(AccessDatabase)
    bpy.utils.unregister_class(DataBase_UIList)
    bpy.utils.unregister_class(DocumentItem)
    bpy.utils.unregister_class(DeleteFromDatabase)
    bpy.utils.unregister_class(ImportFromDataBase)
    del bpy.types.Scene.my_document_collection
    del bpy.types.Scene.my_document_index

    # db test connection and image prep
    # Tests
    bpy.utils.unregister_class(PlaceMesh)
    bpy.utils.unregister_class(ExportToDatabase)
    bpy.utils.unregister_class(DataBaseUIMenu)
    bpy.utils.unregister_class(DataBaseRegister)
    bpy.utils.unregister_class(DataBaseLogin)
    #bpy.utils.unregister_class(StMTestImagePrep)
    #bpy.utils.unregister_class(StMTestConnectionOperator)
    #bpy.utils.unregister_class(StMTestSaveFileToDb)
    #bpy.utils.unregister_class(StMTestGetFileFromDbFromUserId)
    #bpy.utils.unregister_class(StMTestDeleteFileFromDbFromUserId) 

if __name__ == "__main__":
    register()