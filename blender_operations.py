import bpy
import os
import cv2
import io
import tempfile
from .image_processing import PlaneItem
from .file_conversion import blend_opener, fbx_opener
from .DepthByColor import GenerateEdges, NormaliseData, GenerateShapeEdges, GetDistanceBetweenPoints, ColorCheck

#saveObj
#Description
#Saves the Object to a file that can be sent ot the database

#Returns
#filepathAndName: A tuple contianing the filepath and the name of the file
def saveObj():
    filepath = os.path.abspath("ExportFolder\\" + bpy.context.scene.FileName_Input + ".fbx"  )
    bpy.ops.object.select_all()
    bpy.ops.export_mesh.stl(filepath=filepath,  check_existing=True, use_selection=True)
    filepathAndName = (filepath, os.path.basename(filepath) )
    return filepathAndName

#GetLineOfPixels
#Description
#This funciton creates the list of pixels beased on the green pixel indicated
#This list is then inputted into a dictionary with the side (interger} as the key and the list of points as the value

#Parameters
#Polycount: is the amount distance between each found pixel
#ColorWeAreLookingFor: is the color we are trying to find in the image
#Plane: This holds the filepaths and image infomation for the image the user wants to recreate

#Returns
#ImageDictionary: Holds the pixels formingside of the image.  Dict format: Key: Value of the Side. Value: The list of points in the side
#ImageData: The shape of the image (width and length)
#Image: The image that we want to create the mesh off of

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
    ImageData = [ImageRow, ImageColumn]

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
    return ImageDictionary , ImageData , Image

#FindPixels 
#Description: Finds green pixels depending on the side selected above. 
#This funciton will get every pixel a polycount away and then move to the line where we can select a pixel
#if is vertical is True we will get a line of pixels going from the bottom up inside of side to side
#remember that when we talk to the image it flips the image

#Parameters
#Polycount: is the amount distance between each found pixel
#ImageRow: X size of the image (ImageRow, ImageColumn)
#ImageColumn: Y size of the image (ImageRow, ImageColumn)
#Base: is the value that either Row(if we are not vertical) or Column (column if we are vertical) are reset to.
#IteratorValue: is how much the iterator is increased by every loop
#ImageArea: is amount of pixels in the image.
#Image: is the image we are looping throuh
#Color: is the color we are trying to find in the image
#isVertical: determines if we are using the column(isVertical == True) to go through the list of the row(isVertical == False)

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
        if ColorCheck(image, (row, column), Color):
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

#GetLineOfPixels
#Description
#Goes through each line of Pixels present in the image and collects the colors of each pixels
#These colors will be sotred in a dictionary with the coordinate of the Pixel being the Key and the AverageColor of the Pixel being the value

#Parameters
#SideList:the list of pixels from one side of the ImageDictionary. This  should be the right side of the image
#imagedata: provides us with the width and length of the image
#image: provides the image so we can access points on the image 
#ColorToLookOutFor: the color openCv uses as a indicator

#Returns
#PixelLineColorDictionary: This contains the pixels accessed. The key is the Coordinate of a Pixel and the Value is the AverageColor of the Pixels
def GetLineOfPixels(RightSideList, imagedata, image, ColorToLookOutFor):
    LookForPixels = True
    PixelLineColorDictionary = {}
    for Pixel in RightSideList:
        CurrPixel = (Pixel[0]+1, Pixel[1])#this will set us to the next point we are looking for
        while not(CurrPixel[0] == imagedata[0]):
            ImageColor = ColorLister(image, CurrPixel)
            if ImageColor == ColorToLookOutFor: 
                if LookForPixels:
                    PixelLineColorDictionary[CurrPixel] = [0] #we want to record the pixel but we want to set it at 0 because we are at the end of the line
                    LookForPixels = False # we Stop recording the pixels and wait till we either get to the end of the picture or we hit another
                else: LookForPixels = True
            elif LookForPixels: PixelLineColorDictionary[CurrPixel] = round((ImageColor[0]+ ImageColor[1] + ImageColor[2]) /3)
            CurrPixel = (CurrPixel[0]+1, CurrPixel[1])
    return PixelLineColorDictionary

#DefineDictioniesForColorsLines
#Description
#Goes through the Dictionary of points and turns then into a dictionary of lines.
#This Dictionary is defined with it key being the id of the line(interger that is not repeated) and the value being the line that is stored

#Parameters:
#PixelLineColorDictionary: This dictionary contains the coordinates and the ColorAverages that will be converted into Z values

#Returns:
#MeshStructureLibrary: This is a dictionary of MeshStructures which are dictionaries which contain the points Blender needs to generate an image
def DefineDictioniesForColorsLines(PixelLineColorDictionary):
    PixelLinePoints = {}
    MeshStructureLibrary = []
    PixelLinesLibrary = {}
    PixelList = []
    PixelLineList = []
    pixelAsListButActuallyAList = []
    first = True

    iter = 0
    for pixelPoints in PixelLineColorDictionary:
        if first: CurrPixelLines = pixelPoints; first = False #sets the CurrPixelLines
        if CurrPixelLines[1] == pixelPoints[1]:  
            PixelList.append([pixelPoints[0], pixelPoints[1], PixelLineColorDictionary[pixelPoints]]) #if we are on the same line
        else:
            PixelList[-1] = (PixelList[-1][0], PixelList[-1][1], 0) #sets the point at the 
            PixelLinePoints[iter] = PixelList
            PixelList = [] #Reset the PixelList
            iter= iter + 1
            PixelList.append([pixelPoints[0], pixelPoints[1], 0])
            CurrPixelLines = pixelPoints
    
    XValues = []; YValues = [];  ZValues = []
    for PixelLines in PixelLinePoints:
        first =True
        for pixelPoints in PixelLinePoints[PixelLines]:
            if first:YValues.append(pixelPoints[1]); first = False
            XValues.append(pixelPoints[0])  
            ZValues.append(pixelPoints[2]) 
        NormalisedXValues = NormaliseData(XValues)#normalises the X data
        NormalisedZValues = NormaliseData(ZValues)#normalises the Z data

        iterator = 0
        for points in NormalisedXValues:
            PixelLineList.append((points, iterator ,NormalisedZValues[iterator]))
            iterator = iterator + 1
        PixelLinesLibrary[PixelLines] = PixelLineList
        #Resets the list for the next values
        XValues = []
        ZValues = []

    NormalisedYValues = NormaliseData(YValues) #normalises the Y data
    
    iteratorMarkTwo = 0
    for PixelLines in PixelLinesLibrary:
        for pixels in PixelLinesLibrary[PixelLines]: 
            pixelAsList = list(pixels) 
            pixelAsList = pixelAsList[0], NormalisedYValues[iteratorMarkTwo], pixelAsList[2]
            pixelAsListButActuallyAList.append(pixelAsList)
        iteratorMarkTwo = iteratorMarkTwo + 1
    MeshStructureLibrary.append(GenerateEdges(pixelAsListButActuallyAList, "BlenderPoints")) 

    return MeshStructureLibrary

#ColorLister
#Description
#Takes a image and a Pixel's coordinate and finds the corrosponding pixel color for that coordinate

#Parameters
#image: We need the reference image that has the coordinate in it
#PixelCoordinate: This is coordinate being referenced when we get the color from the image.

#Returns
#StandardizedColor: This is the color from the image, but in a list with all of the rgb values set to ints
def ColorLister(image, PixelCoordinate):
    StandardizedColor = [int(image[PixelCoordinate][0]), int(image[PixelCoordinate][1]), int(image[PixelCoordinate][2])]
    return StandardizedColor

#SpaceOutPixels
#Description
#This spaces out the pixels depending on the polyCount

#Parameters
#ImageDictionary: Holds the pixels formingside of the image.  Dict format: Key: Value of the Side. Value: The list of points in the side
#Polycount is the amount distance between each found pixel

#Returns
#FullVertList: This is the updated ImageDictionary contained the spaced out points
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


#NormaliseVertList
#Description
#This normalises the ImageDictionary. This is meant to be here. There is another funciton that does the same function with a list
#We might have to merge the two functions together

#Parameters
#FullVertList:  This is the updated ImageDictionary contained the spaced out points

#Returns
#NewNarrowList: This is the new normalise list used for mesh creation
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

#DrawMeshToScreen
#Description
#This draws the mesh to the blender scene

#Parameters
#ImageDictionary: Holds the pixels formingside of the image.  Dict format: Key: Value of the Side. Value: The list of points in the side
#self: used to report failure
#CollectionName: this is used to name the collection of the shape we draw to the screen

def DrawMeshToScreen(MeshStructure, self, CollectionName = "Sketch_to_Mesh_Collection"):
    if MeshStructure == False: self.report({'ERROR'}, "Invalid Image") 
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
        new_collection = bpy.data.collections.new(CollectionName)
        bpy.context.scene.collection.children.link(new_collection)
        # add object to scene collection
        new_collection.objects.link(new_object)


#DrawMesh
#Description
#This draws the mesh to the blender scene

#Parameters
#ColorWeAreLookingFor: is the color we are trying to find in the image
#Polycount: is the amount distance between each found pixel
#PlaneArray: This is the list of planes used for mesh creation
#isComplex: thhis tells us if we are making a complex image or not

def DrawMesh(ColorWeAreLookingFor, PolyCount, self, PlaneArray:list[PlaneItem], isComplex):
    for plane in PlaneArray:
        ImageDictionary, Imagedata, Image = GetlistOfPixels(PolyCount, ColorWeAreLookingFor, plane)
        #FullVertList = SpaceOutPixels(ImageDictionary, PolyCount)

        if isComplex == True: #only happens when complex is called
            VertList = NormaliseVertList(ImageDictionary)

            if VertList == False:  return False #ends the function before any extra work is done
            MeshStructure = GenerateEdges(VertList, "BlenderPoints")

            DrawMeshToScreen(MeshStructure, self, "Sketch_To_Mesh_Collection") # this is the outline of the mesh
            PixelLineDictionary= GetLineOfPixels(ImageDictionary[0], Imagedata, Image, ColorWeAreLookingFor)
            MeshStructureLibrary = DefineDictioniesForColorsLines(PixelLineDictionary)
            DrawMeshToScreen(MeshStructureLibrary[0], self, "Sketch_To_Mesh_Collection")
        else: 
            MeshStructure = GenerateShapeEdges(ImageDictionary, PolyCount, plane, ColorWeAreLookingFor) #only called when not complex is called
            DrawMeshToScreen(MeshStructure, self) #draws all the meshes to screen
           
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
