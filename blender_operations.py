import bpy
import os
import math
import cv2
import io
import tempfile
from .image_processing import PlaneItem
from .file_conversion import blend_opener, fbx_opener
from .DepthByColor import GenerateEdges, NormaliseData, GenerateShapeEdges

def saveObj():
    filepath = os.path.abspath("ExportFolder\\" + bpy.context.scene.FileName_Input + ".fbx"  )
    bpy.ops.object.select_all()
    bpy.ops.export_mesh.stl(filepath=filepath,  check_existing=True, use_selection=True)
    filepathAndName = (filepath, os.path.basename(filepath) )
    return filepathAndName

def GetlistOfPixels(ColorWeAreLookingFor, plane:PlaneItem): #(0, 255, 0) # Green at the moment.
    ImageDictionary = {}
    Image = cv2.imread(plane.ImagePlaneFilePath)
    for iterator in range(4):
        match iterator: # this will loop through the image and gather the green pxiels outlined on each side
            case 0: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="Right")
            case 1: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="Left")
            case 2: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="MiddleRight")
            case 3: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="MiddleLeft")
    return ImageDictionary

def DefinePixels(image, Color, Direction):
    PixelsList: list = []
    imagedata = [image.shape[0], image.shape[1]]

    #we want to cut the image in half and have one case go one way and the other go the opposite way
    halfImageDataY = int(imagedata[0] * 0.5) if imagedata[0] % 2 == 0 else int(imagedata[0] * 0.5)
    halfImageData = ( halfImageDataY, imagedata[1])

    match Direction:
        case "Right": CurrRangePair = (range(imagedata[1] -1), range(imagedata[0]-1))
        case "Left": CurrRangePair = (range(imagedata[1] -1), range(imagedata[0] -1))
        case "MiddleRight" : CurrRangePair = (range(halfImageData[1] -1), range(halfImageData[0]- 1 ))
        case "MiddleLeft": CurrRangePair = (range(halfImageData[1] -1), range(imagedata[0] - halfImageData[0]-  1 ))

    row = 0; column = 0
    if (Direction == "Right" or Direction == "MiddleRight"):
        base = (imagedata[0] - halfImageData[0] - 1) if  Direction == "MiddleRight" else 0
        for i in CurrRangePair[0]:
            for j in CurrRangePair[1]:
                if row == imagedata[1]: break
                elif column == imagedata[0]: row = row + 1; column = base   # if we get to the end of the row, row is increase to get to the next row and column is reset
                elif (int(image[column, row][0]), int(image[column, row][1]), int(image[column, row][2])) == Color: # if we find the color green 
                    PixelsList.append((column, row)) # if we find a pixel we are looking for we add it to the pixel List
                    row = row + 1; column = base  #row is increase to get to the next row and column is reset
                elif row * column >= (imagedata[0] + 1) * (imagedata[1]+ 1): break
                column = column + 1

    elif (Direction == "Left" or Direction == "MiddleLeft"): 
        base = (imagedata[0] - halfImageData[0] - 1) if  Direction == "MiddleLeft" else 0
        for i in CurrRangePair[0]:
            for j in reversed(CurrRangePair[1] ):
                if row == imagedata[1]: break # we need to take the absoulte value of the column alot through this loop
                elif abs(column) == imagedata[0]: row = row + 1; column = base # if we get to the end of the row, row is increase to get to the next row and column is reset
                elif (int(image[abs(column), row][0]), int(image[abs(column), row][1]), int(image[abs(column), row][2])) == Color: # if we find the color green 
                    PixelsList.append((abs(column), row)) # if we find a pixel we are looking for we add it to the pixel List
                    row = row + 1; column = base  #row is increase to get to the next row and column is reset
                elif abs(row * column) >= abs((imagedata[0] + 1) * (imagedata[1]+ 1)): break
                column = column - 1
                
    return PixelsList

def SpaceOutPixels(ImageDictionary, PolyCount):
    FullVertList = {} #new dictionary to 

    for Sides in ImageDictionary:
        NextIter = 1 # keeps count of the value we are on, so we can get the next value in the list
        CurrIter = 0
        VertList: list = [] #the vertList saves the verts that are out of the poly count distance. The vert list is kept here so it will be reset fro each side
        done = False
        while not done:
            ImageDictIter = (ImageDictionary[Sides]) # creates a varible to short formulas
            XDistance = abs((ImageDictIter[NextIter][0] - ImageDictIter[CurrIter][0])^2) # gets the X part of the Distance formula
            YDistance = abs(((ImageDictIter[NextIter])[1] - ImageDictIter[CurrIter][1])^2) # gets the Y part of the Distance formula

            if (NextIter == ImageDictIter.__len__() -1):
                VertList.append(ImageDictIter[NextIter]) # we add the last vertex to the list 
                done = True; break # sets done to true so the while loop will end
            
            elif math.sqrt(XDistance + YDistance) >= PolyCount : 
                VertList.append(ImageDictIter[NextIter]) # we save the next vertex into the VertList
                CurrIter = NextIter # we get to the new vertex so we can find the point after

            NextIter = NextIter + 1        
        FullVertList[Sides] = VertList
    return FullVertList
    
def GetZAxisByColor(FullVertList, PolyCount, plane:PlaneItem):
    return GenerateShapeEdges(FullVertList, PolyCount, plane )#polycount is our radius


def NormaliseVertList(FullVertList):
    xArray = []; yArray = []

    #since we know that there are only are only 4 sides in the FullVertList we can add all of th points into a new list
    for VertList in FullVertList:
        for verts in FullVertList[VertList]: # we separate the X and Y values so we can normalise the data sets
            xArray.append(verts[0]) ; yArray.append(verts[1])

    # normalising the arrays 
    xArray = NormaliseData(xArray); yArray = NormaliseData(yArray)
    if xArray == False or yArray == False: return False# if any of the arrays are empty

    # we then take the separted normal data input them back into coordinates and add them to the list
    NarrowedNormalisedVertList = []
    for count in range(xArray.__len__()): NarrowedNormalisedVertList.append(((xArray[count]), (yArray[count]), (1.0))) # we add the one into the list so the list will have the Z coordinate.
    NewNarrowList = tuple(NarrowedNormalisedVertList)

    return NewNarrowList   

def CreateEdges(VertList):
    if VertList == False:  return False #ends the function before any extra work is done

    MeshStructure = GenerateEdges(VertList, "BlenderPoints")
    MeshStructure[2] = [] #this will hold the faces
    return MeshStructure

def DrawMeshToScreen(MeshStructure, self):
    if MeshStructure == False:
        self.report({'ERROR'}, "Invalid Image") 
    else:
        # make mesh
        new_mesh = bpy.data.meshes.new('new_mesh')
        new_mesh.from_pydata(MeshStructure[0], MeshStructure[2], MeshStructure[1])
        new_mesh.update()

        if not new_mesh.edges:
            edges = MeshStructure[1] # Define edges here based on your vertices
            new_mesh.from_pydata(MeshStructure[0], edges, MeshStructure[1])
            new_mesh.update()

        # make object from mesh
        new_object = bpy.data.objects.new('Sketch_to_Mesh_mesh', new_mesh)
        # make collection
        new_collection = bpy.data.collections.new('Sketch_to_Mesh_Collection')
        bpy.context.scene.collection.children.link(new_collection)
        # add object to scene collection
        new_collection.objects.link(new_object)

#Draws all the non Complex meshes to screen 
def DrawMeshesToScreen(ColorWeAreLookingFor, PolyCount, self, PlaneArray:list[PlaneItem], isComplex):
    for plane in PlaneArray:
        ImageDictionary = GetlistOfPixels(ColorWeAreLookingFor, plane)
        FullVertList = SpaceOutPixels(ImageDictionary, PolyCount)

        if isComplex == True: #only happens when complex is called
            VertList = NormaliseVertList(FullVertList)
            MeshStructure = CreateEdges(VertList)
        else:  MeshStructure = GetZAxisByColor(FullVertList, PolyCount * 10, plane) #only called when not complex is called

        #draws all the meshes to screen
        DrawMeshToScreen(MeshStructure, self)

# TODO: return something that is not 0. case handling and error handling, as well as completed and noncompleted states.
def encode_file(file_path):
    
   with open(file_path, "rb") as file:
        blend_file_contents = io.BytesIO(file.read())
        return blend_file_contents

# TODO: return something that is not 0. case handling and error handling, as well as completed and noncompleted states.
def decode_file(data, file_extension):
    #Apparently the data doesn't need to be decoded so we will handle the different
    #file extensions handled here instead of outside the file_conversion.py file

    #write the data into a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) #we'll probably have to add another parameter here for the file extension or soemthing else)
    temp_file.write(data)
    temp_file.close()

    #Deal with the separate file extensions
    match file_extension :
        case ".blend":
            blend_opener(temp_file.name)
        case ".fbx":
            fbx_opener(temp_file.name)
        case _: #defualt case # if there is an image file
            bpy.ops.import_image.to_plane(files=[{"name":temp_file.name, "name":temp_file.name}], directory="", relative=False)

    #remove the temp file
    os.unlink(temp_file.name)

    #if we are returning just the file back then cases checking will have to happen outside of this method
    return 0

