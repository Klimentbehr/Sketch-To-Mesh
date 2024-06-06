bl_info = {
    "name": "Sketch_To_MeshV2",
    "author": "LuckyNinjas",
    "version": (0, 0, 2),
    "blender": (4, 1, 1),
    "location": "3D Viewport > Sidebar > My Custom Panel category",
    "description": " This application's purpose is to convert a sketch or drawing into a mesh that can be imported into Blender. The application will take a set of images in .png extensions and run those images through a machine-learning model to output a usable mesh. This mesh will then be exported so that the user can do what they want with the file, such as copying and sharing the file or personally using it.",
    "category": "Development",
    "wheels" : [
        "./wheels/absl_py-2.1.0-py3-none-any.whl",
        "./wheels/astunparse-1.6.3-py2.py3-none-any.whl",
        "./wheels/bcrypt-4.1.3-cp39-abi3-macosx_10_12_universal2.whl",
        "./wheels/bcrypt-4.1.3-cp39-abi3-win_amd64.whl",
        "./wheels/certifi-2024.6.2-py3-none-any.whl",
        "./wheels/charset_normalizer-3.3.2-cp312-cp312-win_amd64.whl",
        "./wheels/dnspython-2.6.1-py3-none-any.whl",
        "./wheels/flatbuffers-24.3.25-py2.py3-none-any.whl",
        "./wheels/gast-0.5.4-py3-none-any.whl",
        "./wheels/google_pasta-0.2.0-py3-none-any.whl",
        "./wheels/grpcio-1.64.1-cp312-cp312-win_amd64.whl",
        "./wheels/h5py-3.11.0-cp312-cp312-win_amd64.whl",
        "./wheels/idna-3.7-py3-none-any.whl",
        "./wheels/keras-3.3.3-py3-none-any.whl",
        "./wheels/libclang-18.1.1-py2.py3-none-win_amd64.whl",
        "./wheels/markdown_it_py-3.0.0-py3-none-any.whl",
        "./wheels/Markdown-3.6-py3-none-any.whl",
        "./wheels/MarkupSafe-2.1.5-cp312-cp312-win_amd64.whl",
        "./wheels/mdurl-0.1.2-py3-none-any.whl",
        "./wheels/ml_dtypes-0.3.2-cp312-cp312-win_amd64.whl",
        "./wheels/namex-0.0.8-py3-none-any.whl",
        "./wheels/numpy-1.26.4-cp312-cp312-macosx_11_0_arm64.whl",
        "./wheels/numpy-1.26.4-cp312-cp312-win_amd64.whl",
        "./wheels/opencv_python-4.9.0.80-cp37-abi3-macosx_11_0_arm64.whl",
        "./wheels/opencv_python-4.9.0.80-cp37-abi3-win_amd64.whl",
        "./wheels/opt_einsum-3.3.0-py3-none-any.whl",
        "./wheels/optree-0.11.0-cp312-cp312-win_amd64.whl",
        "./wheels/packaging-24.0-py3-none-any.whl",
        "./wheels/protobuf-4.25.3-cp310-abi3-win_amd64.whl",
        "./wheels/pygments-2.18.0-py3-none-any.whl",
        "./wheels/pymongo-4.6.1-cp312-cp312-win_amd64.whl",
        "./wheels/requests-2.32.3-py3-none-any.whl",
        "./wheels/rich-13.7.1-py3-none-any.whl",
        "./wheels/setuptools-70.0.0-py3-none-any.whl",
        "./wheels/six-1.16.0-py2.py3-none-any.whl",
        "./wheels/tensorboard_data_server-0.7.2-py3-none-any.whl",
        "./wheels/tensorboard-2.16.2-py3-none-any.whl",
        "./wheels/tensorflow_intel-2.16.1-cp312-cp312-win_amd64.whl",
        "./wheels/termcolor-2.4.0-py3-none-any.whl",
        "./wheels/typing_extensions-4.12.1-py3-none-any.whl",
        "./wheels/urllib3-2.2.1-py3-none-any.whl",
        "./wheels/werkzeug-3.0.3-py3-none-any.whl",
        "./wheels/wheel-0.43.0-py3-none-any.whl",
        "./wheels/wrapt-1.16.0-cp312-cp312-win_amd64.whl",
        "./wheels/pillow-10.3.0-cp312-cp312-win_amd64.whl",
    ]
}

import bpy

from .ui_operations import OBJECT_OT_add_plane_item, Reset_Input_Images,VIEW3D_PT_Sketch_To_Mesh_Views_FilePath_Panel, PlaceImageIn3D, PlaceMesh, NotificationPopup, VIEW3D_PT_Sketch_To_Mesh_MeshSettings_Panel, Reset_Mesh_Collection, PredictAndPlace, StMTestCameraDetectionAI, StMTestCameraDetection
from .testing_operations import DoImg, ExportToDatabase#, , StMTestImagePrep, StMTestSaveFileToDb, StMTestConnectionOperator, StMTestGetFileFromDbFromUserId, StMTestDeleteFileFromDbFromUserId,
from .DatabaseUI import DataBaseLogin, DataBaseRegister, DocumentItem, DataBaseLogout,  DataBase_UIList, DeleteFromDatabase, AccessDatabase, AddToDatabase, ImportFromDataBase
from .base_ui import VIEW3D_PT_Sketch_To_Mesh_Panel, DataBaseUIMenu #, VIEW3D_PT_Sketch_To_Mesh_Testing


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
    bpy.types.Scene.PlaneRotation = bpy.props.EnumProperty(name="PlaneRotation", description="Roations",items=[("front", "front", "front"), ("Right Side", "Right Side", "Right Side"), ("Back","Back","Back"), ("Left Side", "Left Side", "Left Side"), ("Top", "Top", "Top"), ("Bottom","Bottom","Bottom")])

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
    
    # model and camera
    bpy.utils.register_class(PredictAndPlace)
    bpy.utils.register_class(StMTestCameraDetection)
    bpy.utils.register_class(StMTestCameraDetectionAI)
    

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
    
    # model
    bpy.utils.unregister_class(PredictAndPlace)
    bpy.utils.unregister_class(StMTestCameraDetection)
    bpy.utils.unregister_class(StMTestCameraDetectionAI)

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