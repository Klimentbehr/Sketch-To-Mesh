import bpy
import os
import math
import cv2
from .image_processing import PlaneItem


def saveObj():
    filepath = os.path.abspath("ExportFolder\\TempExport.fbx")

    bpy.ops.object.select_all()
    bpy.ops.export_mesh.stl(filepath=filepath,  check_existing=True, use_selection=True)
    filepathAndName = (filepath, os.path.basename(filepath) )
    return filepathAndName

def GetlistOfPixels(ColorWeAreLookingFor, plane:PlaneItem): #(0, 255, 0) # Green at the moment.
    """
    The Dictionary below is used to hold the placement of pixels positions
    The key for this dictionary is the "side" the pixel is located on
    Example for the dataset is: {0: [(176,142), (175, 143)...]}
    """
    ImageDictionary = {} #Dictionary that holds the placement of the pixels, the side is the key 
    Image = cv2.imread(plane.ImagePlaneFilePath)
    for iterator in range(5):
        match iterator: # this will loop through the image and gather the green pxiels outlined on each side
            case 0: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="Right")
            case 1: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="Left")
            case 2: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="MiddleRight")
            case 3: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="MiddleLeft")
            case 4: ImageDictionary[iterator] = DefinePixels(image=Image, Color=ColorWeAreLookingFor, Direction="Vertical")
    return ImageDictionary

def DefinePixels(image, Color, Direction):
    PixelsList: list = []
    PixelIter = 0 #iterator will hold the position of the previous pixel
    imagedata = [image.shape[0], image.shape[1]]
    #imagedata[0] is column or y values
    #imagedata[1] is row or x values

    #we want to cut the image in half and have one case go one way and the other go the opposite way
    halfImageDataY = int(imagedata[0] * 0.5) if imagedata[0] % 2 == 0 else int(imagedata[0] * 0.5)
    halfImageData = ( halfImageDataY, imagedata[1])

    match Direction:
        case "Right": CurrRangePair = (range(imagedata[1] -1), range(imagedata[0]-1))
        case "Left": CurrRangePair = (range(imagedata[1] -1), range(imagedata[0] -1))
        case "MiddleRight" : CurrRangePair = (range(halfImageData[1] -1), range(halfImageData[0]- 1 ))
        case "MiddleLeft": CurrRangePair = (range(halfImageData[1] -1), range(imagedata[0] - halfImageData[0]-  1 ))
        case "Vertical": CurrRangePair =  (range(imagedata[1] -1), range(imagedata[0]-1))
    row = 0; column = 0
    if (Direction == "Right" or Direction == "MiddleRight"):
        base = (imagedata[0] - halfImageData[0] - 1) if  Direction == "MiddleRight" else 0
        for i in CurrRangePair[0]:
            for j in CurrRangePair[1]:
                if row == imagedata[1]: break
                elif column == imagedata[0]: row = row + 1; column = base   # if we get to the end of the row, row is increase to get to the next row and column is reset
                elif (int(image[column, row][0]), int(image[column, row][1]), int(image[column, row][2])) == Color: # if we find the color green 
                    if (PixelsList.__len__ == 0): PixelsList.append((column, row)) #If the pixel list is empty then it is the first pixel to be added
                    else: #any subsequent pixels will need to be checked to make sure that they are within acceptable distance from one another
                        if (abs(PixelsList[PixelIter][0] - column) < 50 or (abs(PixelsList[PixelIter][1] - row) < 50)):
                            PixelsList.append((column, row)) #if we find a pixel we are looking for we add it to the pixel List
                            row = row + 1; column = base  #row is increase to get to the next row and column is reset
                            PixelIter = PixelIter + 1 #set the PixelIter to the pixel that was just added
                        else: continue
                elif row * column >= (imagedata[0] + 1) * (imagedata[1]+ 1): break
                column = column + 1

    elif (Direction == "Left" or Direction == "MiddleLeft"): 
        base = (imagedata[0] - halfImageData[0] - 1) if  Direction == "MiddleLeft" else 0
        for i in CurrRangePair[0]:
            for j in reversed(CurrRangePair[1] ):
                if row == imagedata[1]: break # we need to take the absoulte value of the column alot through this loop
                elif abs(column) == imagedata[0]: row = row + 1; column = base # if we get to the end of the row, row is increase to get to the next row and column is reset
                elif (int(image[abs(column), row][0]), int(image[abs(column), row][1]), int(image[abs(column), row][2])) == Color: # if we find the color green 
                    if (PixelsList.__len__ == 0): PixelsList.append((abs(column), row)) #If the pixel list is empty then it is the first pizel to be added
                    else: #any subsequent pixels will need to be checked to make sure that they are within acceptable distance from one another
                        if (abs(PixelsList[PixelIter][0] - column) < 50 or (abs(PixelsList[PixelIter][1] - row) < 50)):
                            PixelsList.append((abs(column), row)) # if we find a pixel we are looking for we add it to the pixel List
                            row = row + 1; column = base  #row is increase to get to the next row and column is reset
                            PixelIter = PixelIter + 1 #set the PixelIter to the pixel that was just added
                        else: continue
                elif abs(row * column) >= abs((imagedata[0] + 1) * (imagedata[1]+ 1)): break
                column = column - 1

    elif (Direction == "Vertical"):
        base = (imagedata[0] - halfImageData[0] - 1)
        for i in CurrRangePair[1]:
            for j in CurrRangePair[0]:
                if column == imagedata[0]: break #since we are doing vertical and not horizontal we need to check for column and not row
                elif row == imagedata[1]: column = column + 1; row = base #again this is the inverse since it is vertical
                elif (int(image[column, row][0]), int(image[column, row][1]), int(image[column, row][2])) == Color: #color checking is the same
                    if (PixelsList.__len__ == 0): PixelsList.append((column, row)) #If the pixel list is empty then it is the first pixel to be added
                    else: #any subsequent pixels will need to be checked to make sure that they are within acceptable distance from one another
                        if (abs((PixelsList[-1])[0] - column) < 50 or (abs((PixelsList[-1])[1] - row) < 50)):
                            PixelsList.append((column, row)) # if we find a pixel we are looking for we add it to the pixel List
                            column = column + 1; row = base #column is increased to get to the next row and column
                            PixelIter = PixelIter + 1 #set the PixelIter to the pixel that was just added
                        else: continue
                elif row * column >= ((imagedata[0] + 1) * (imagedata[1] + 1)): break
                row = row + 1

                
    return PixelsList

def SpaceOutPixels(ColorWeAreLookingFor, PolyCount, plane:PlaneItem):
    ImageDictionary = GetlistOfPixels(ColorWeAreLookingFor, plane)
    FullVertList = {} #new dictionary that spaces out the points

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
    
def NormaliseVertList(ColorWeAreLookingFor, PolyCount, plane):
    FullVertList = SpaceOutPixels(ColorWeAreLookingFor, PolyCount, plane)
    xArray = []; yArray = [] #These two arrays will hold the positions of the x points and the y values respectively

    #since we know that there are only are only 4 sides in the FullVertList we can add all of the points into a new list
    for VertList in FullVertList:
        for verts in FullVertList[VertList]: # we separate the X and Y values so we can normalise the data sets
            xArray.append(verts[0]) ; yArray.append(verts[1])

    # normalising the arrays 
    xArray = NormaliseData(xArray); yArray = NormaliseData(yArray)
    if xArray == False or yArray == False: return False# if any of the arrays are empty

    # we then take the separted normal data input them back into coordinates and add them to the list
    NarrowedNormalisedVertList = []
    for count in range(xArray.__len__()): NarrowedNormalisedVertList.append(((xArray[count]), (yArray[count]), (1.0))) # we add the one into the list so the list will have the Z coordinate.
    #Blender doesn't like dictionaries so we have to create a tuple in order to store the X,Y, and Z coordinates
    NewNarrowList = tuple(NarrowedNormalisedVertList)
    # sortedNarrowList = OrderVertMesh(NewNarrowList)
    return NewNarrowList

def NormaliseData(List:list):
    #This list sets the data to either 0 or 1
    NewList = []
    if not List: return False
    else: 
        for element in List:  
            norm = (element - min(List)) / (max(List) - min(List))
            NewList.append(norm)
    return NewList

def CreateEdges(ColorWeAreLookingFor, PolyCount, plane:PlaneItem):
    MeshStructure = {}
    VertList = NormaliseVertList(ColorWeAreLookingFor, PolyCount, plane)
    if VertList == False:  return False #ends the function before any extra work is done

    MeshStructure[0] = VertList
    edgeList:list =[]
    iterator = 1

    for verts in range(VertList.__len__()-1): #this will get the vertical edges for the mesh.
        if iterator >=  VertList.__len__()-1: 
            edgeList.append((0, iterator))
            # iterator = iterator + 1
        else:
            edgeList.append((verts, iterator))
            iterator = iterator + 1
    MeshStructure[1] = edgeList
    # We also need to get the horizontal. But maybe later
    MeshStructure[2] = [] #this will hold the faces
    return MeshStructure

def DrawMeshToScreen(ColorWeAreLookingFor, PolyCount, self, plane:PlaneItem):
    MeshStructure = CreateEdges(ColorWeAreLookingFor, PolyCount, plane)
    if MeshStructure == False:
        self.report({'ERROR'}, "Invalid Image") 
    else:
        # make mesh
        new_mesh = bpy.data.meshes.new('new_mesh')
        
        new_mesh.from_pydata(MeshStructure[0], MeshStructure[2], MeshStructure[1])
        new_mesh.update()
        # make object from mesh
        new_object = bpy.data.objects.new('Sketch_to_Mesh_mesh', new_mesh)
        # make collection
        new_collection = bpy.data.collections.new('Sketch_to_Mesh_Collection')
        bpy.context.scene.collection.children.link(new_collection)
        # add object to scene collection
        new_collection.objects.link(new_object)

def DrawAllMeshesToScreen(ColorWeAreLookingFor, PolyCount, self, PlaneArray:list[PlaneItem]):
    for plane in PlaneArray:
        DrawMeshToScreen(ColorWeAreLookingFor, PolyCount, self, plane)

def CheckSpaceBetweenPoints(X2, X1, Y2, Y1):
    return math.sqrt(((X2-X1)^2) + ((Y2-Y1)^2))