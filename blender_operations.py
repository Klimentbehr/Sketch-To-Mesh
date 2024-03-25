import bpy
import os
import cv2
import io
import tempfile
from .image_processing import PlaneItem
from .file_conversion import blend_opener, fbx_opener
from .DepthByColor import GenerateEdges, NormaliseData, GenerateShapeEdges, GetDistanceBetweenPoints

def saveObj():
    filepath = os.path.abspath("ExportFolder\\" + bpy.context.scene.FileName_Input + ".fbx"  )
    bpy.ops.object.select_all()
    bpy.ops.export_mesh.stl(filepath=filepath,  check_existing=True, use_selection=True)
    filepathAndName = (filepath, os.path.basename(filepath) )
    return filepathAndName

def GetlistOfPixels(PolyCount, ColorWeAreLookingFor, plane:PlaneItem): #(0, 255, 0) # Green at the moment.
    """
    The Dictionary below is used to hold the placement of pixels positions
    The key for this dictionary is the "side" the pixel is located on
    Example for the dataset is: {0: [(176,142), (175, 143)...]}
    """
    ImageDictionary = {} #Dictionary that holds the placement of the pixels, the side is the key 
    Image = cv2.imread(plane.ImagePlaneFilePath)
    ImageRow = Image.shape[0] - 1
    ImageColumn = Image.shape[1] -1
    HalfImageRow = round(ImageRow * 0.5)
    AreaOfImage = ImageRow * ImageColumn

    for iterator in range(5):
        match iterator: # this will loop through the image and gather the green pxiels outlined on each side
            #Right
            case 0: ImageDictionary[iterator] = FindPixels(PolyCount, ImageRow, ImageColumn, base= 0, iteratorValue= 1, ImageArea = AreaOfImage, image = Image, Color=ColorWeAreLookingFor) 
            #Left
            case 1: ImageDictionary[iterator] = FindPixels(PolyCount, ImageRow, ImageColumn, base= ImageRow, iteratorValue= -1, ImageArea = AreaOfImage, image = Image, Color=ColorWeAreLookingFor)
            #Middle Right
            case 2: ImageDictionary[iterator] = FindPixels(PolyCount, ImageRow, ImageColumn, base= HalfImageRow, iteratorValue= 1, ImageArea = AreaOfImage, image = Image, Color=ColorWeAreLookingFor)
            #middle Left
            case 3: ImageDictionary[iterator] = FindPixels(PolyCount, ImageRow, ImageColumn, base= HalfImageRow, iteratorValue= -1, ImageArea = AreaOfImage, image = Image, Color=ColorWeAreLookingFor)
            #Vertical
            case 4: ImageDictionary[iterator] = FindPixels(PolyCount, ImageRow, ImageColumn, base= HalfImageRow, iteratorValue= 1, ImageArea = AreaOfImage, image = Image, Color=ColorWeAreLookingFor, isVertical=True)
    return ImageDictionary

#FindPixels 
#Description: Finds green pixels depending on the side selected above. 
#This funciton will get every pixel a polycount away and then move to the line where we can select a pixel
#if is vertical is True we will get a line of pixels going from the bottom up inside of side to side

#remember that when we talk to the image it flips the image

#Parameters
#Polycount is the amount distance between each found pixel
#ImageRow X size of the image (ImageRow, ImageColumn)
#ImageColumn Y size of the image (ImageRow, ImageColumn)
#Base is the value that either Row(if we are not vertical) or Column (column if we are vertical) are reset to.
#IteratorValue is how much the iterator is increased by every loop
#ImageArea is amount of pixels in the image.
#Image is the image we are looping throuh
#Color is the color we are trying to find in the image
#isVertical determines if we are using the column(isVertical == True) to go through the list of the row(isVertical == False)

#Return
#this function returns the list of pixels found
def FindPixels(PolyCount, ImageRow, ImageColumn, base, iteratorValue, ImageArea, image, Color, isVertical = False):
    PixelList = []
    row = 0; column = 0
    for points in range(ImageArea):
        if abs(row) >= ImageRow:
            if isVertical: break
            else: row = base; column = column + 1 # we reset the row and increament the column
        if abs(column) >= ImageColumn:  # when we get to the end of the image
            if isVertical: column = base; row = row + 1
            else: break
        if (int(image[row, column][0]), int(image[row, column][1]), int(image[row, column][2])) == Color:
            PixelList.append((row, column)) #If the pixel list is empty then it is the first pixel to be added
            if isVertical:
                if (row + PolyCount) > ImageRow : break 
                else: row = row + PolyCount; column = base
            else:
                if (column + PolyCount) > ImageColumn : break 
                else: column = column + PolyCount; row = base
        else:
            if isVertical : column = column + iteratorValue
            else: row = row + iteratorValue
    return PixelList

#Not really useful because we do this in the first function now
def SpaceOutPixels(ImageDictionary, PolyCount):
    FullVertList = {} #new dictionary to 

    for Sides in ImageDictionary:
        VertList: list = [] #the vertList saves the verts that are out of the poly count distance. The vert list is kept here so it will be reset fro each side
        done = False
        ImageDictIter = (ImageDictionary[Sides]) # creates a varible to shorten formulas
        while not done:
            if (1 == len(ImageDictIter)):
                ImageListIter:list = ImageDictIter[0] 
                NextImageListIter:list = ImageDictIter[1] 
                VertList.append(ImageListIter) # we add the last vertex to the list 
                done = True; break # sets done to true so the while loop will end
            
            if (abs(ImageListIter[0] - NextImageListIter[0]) > 50 or (abs(ImageListIter[1] - NextImageListIter[1]) > 50)): ImageDictIter = SearchForClosestPoint(ImageDictIter, NextImageListIter)
            elif GetDistanceBetweenPoints(ImageListIter, NextImageListIter) >= PolyCount: VertList.append(NextImageListIter) # we save the next vertex into the VertList
            del ImageDictIter[0]  
        FullVertList[Sides] = VertList
    return FullVertList
    
def GetZAxisByColor(FullVertList, PolyCount, plane:PlaneItem):
    return GenerateShapeEdges(FullVertList, PolyCount, plane )#polycount is our radius

def NormaliseVertList(FullVertList):
    xArray = []; yArray = []

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
        ImageDictionary = GetlistOfPixels(PolyCount, ColorWeAreLookingFor, plane)
        #FullVertList = SpaceOutPixels(ImageDictionary, PolyCount)

        if isComplex == True: #only happens when complex is called
            VertList = NormaliseVertList(ImageDictionary)
            MeshStructure = CreateEdges(VertList)
        else:  MeshStructure = GetZAxisByColor(ImageDictionary, PolyCount * 10, plane) #only called when not complex is called

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


def SearchForClosestPoint(PointArray, startingPoint ):
    closestDistance = 100 #will be used to check the distance between two points
    tempIter = 0 #will loop thought the for each loop and hold a temp Iterator
    finalIter = 0 #will hold the iterator of the closest point

    for nextPoint in PointArray:
        TempDistance = abs(GetDistanceBetweenPoints(startingPoint, nextPoint)) #get the distance between the 
        if(TempDistance == 0):continue #If the tempDistance ends up being 0 then that means they are the exact same distance 
        elif(TempDistance < closestDistance): #if the distance is shorter that the current shortest distance then change the shortest distance to the new distance
            finalIter = tempIter #also set the finalIter to the shortest distance Iterator
            closestDistance = TempDistance
        tempIter = tempIter + 1

    NewList: list = PointArray[finalIter:] #adjust the list so that the closest points are now first in the list
    NewList.extend(PointArray[:finalIter]) #adjust the list so that the further points are now after the closest points
    return NewList
