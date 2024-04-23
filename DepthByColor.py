import cv2
import math
from .image_processing import PlaneItem, mark_corners, EditPicture, SaveImage
from .DepthByColorHelper import AdjacentEdge, ImageDataClass, GetSlope, calucalateYIntercept, GetUniquePoints, CheckForInsideLines, CreateSolidLine, GetDistanceBetweenPoints, ColorCheck, GetFilledCircle, GetAverageOfAllCoordinateValuesInList, GetAverageDstBetweenPoints, getCircle, GetClosetPointsToValue, GetUniquePoints
from threading import Thread, Lock
from dataclasses import dataclass

mutex = Lock()
meshMidpoint = None

@dataclass
class EdgeData:
    Point = [] #this will hold the coordinate of the center point
    NextPoint = []#this holds the next point in the shape
    slope:float #this holds the slope of the line with the first one and the next point
    Yintercept = [] #this holds the Yintercept of the line between the first and next point
    PointToBeCheckedForColor = []
    AllSurrondingPoints = []
    AverageColor:tuple
    ZValue:int
    LinePoints:list

    def __init__(self, Point ,NextPoint):
        self.Point = Point
        self.NextPoint = NextPoint
        self.slope = GetSlope(Point, NextPoint) #gets the slope of each edge
        self.Yintercept = calucalateYIntercept(Point, self.slope )

#GenerateShapeEdges
#Description
#This function create the edgedata used throughout the process of making the simplified mesh
#At the moment it uses 2 ways of edge detection to get the needed points.
#The first being the manual detection which finds the edges based on distance
#The sceond being the edges openCv has found. (These edges will be used for the first state of the program)

#Parameters
#radius: This is the polycount passed in from blender operations.
#Plane: This holds the filepaths and image infomation for the image the user wants to recreate
#ColorToLLokFor: This is the color openCv uses for outlining

#Returns
#outputlist: This is the MeshStructure of the simplified mesh
def GenerateShapeEdges(radius:int, plane:PlaneItem, ColorToLookFor):
    imageDataClass = ImageDataClass(radius, plane, ColorToLookFor)
    FinishedList =[]
    SizedEdgePoints, MulipliersValues, imageShape = GetPointsFromImage(imageDataClass.image, plane, imageDataClass.ImageShape[0], imageDataClass.ImageShape[1], radius)
    imageDataClass.__setattr__('ImageShape', imageShape)
    imageDataClass.__setattr__('image', cv2.resize(imageDataClass.image, (imageDataClass.ImageShape[1], imageDataClass.ImageShape[0])))
    for points in SizedEdgePoints: FinishedList.append((round(points[0] / MulipliersValues[0]), round(points[1] / MulipliersValues[1])))

    for points in FinishedList:
        EditPicture((0,0,0), points, imageDataClass.image)
        SaveImage(imageDataClass.image, plane.ImagePlaneFilePath, "View0")

    # FinishedDict = {key: 0 for key in FinishedList}

    # AdjacentPoint:AdjacentEdge = CheckForInsideLines(imageDataClass, FinishedDict)
    # for adjacentLines in AdjacentPoint.AdjacentLine: 
    #     for points in AdjacentPoint.AdjacentLine[adjacentLines]: #adds on the extras lines 
    #         EditPicture(imageDataClass.Color, points, imageDataClass.image)
    #         SaveImage(imageDataClass.image, imageDataClass.plane.ImagePlaneFilePath, "View0")
 
    EdgeDataList = CreateEdgeData(FinishedList, imageDataClass)
    EdgeDataList = CalculateLocationsOfAvaliblePixelsAroundPoint(EdgeDataList, imageDataClass)
    outputlist = CycleThroughEdgePointsForColor(EdgeDataList, imageDataClass)
    return outputlist

#GetPointsFromImage
#Description
#This function is gets the part of the edgePoints from OpenCv. 

#Parameters
#image: This is the image generated from the plane information
#plane: This holds the filepaths and image infomation for the image the user wants to recreate
#ImageRow: This is the width of the image
#ImageColumn: This is the Lebngth of the image

#Returns
#This function returns the list of edgePoint coordinates, The Sizers for the row and column of the image, and the size of the oringinal image
def GetPointsFromImage(image, plane:PlaneItem, ImageRow, ImageColumn, radius):
    CvEdgePointArray, imageShape = mark_corners(plane.PlaneFilepath)
    EdgeImageRow = imageShape[0] -1
    EdgeImageColumn = imageShape[1] -1
    EnlargedImageRowMultiplier = ImageRow / EdgeImageRow
    EnlargedImageColumnMultiplier = ImageColumn / EdgeImageColumn
    EdgePointArray = []
    EdgepointsForCircle = []
    EdgePoints=[]

    for point in CvEdgePointArray:
        EnlargedRow = round(point[0] * EnlargedImageRowMultiplier)
        EnlargedColummn = round(point[1] * EnlargedImageColumnMultiplier)
        EdgePointArray.append((EnlargedRow, EnlargedColummn))
        
    AveragePoint = GetAverageOfAllCoordinateValuesInList(EdgePointArray)
    AverageValueTODstCircle = GetAverageDstBetweenPoints(EdgePointArray, AveragePoint)
    CirclePoints = getCircle(AveragePoint, image, AverageValueTODstCircle+round(EnlargedRow), True)
    for CircleEdges in CirclePoints: #CirclePoints in a list of lists so we searchthrough each list to see if there are any edges
        EdgepointsForCircle = EdgepointsForCircle + GetClosetPointsToValue(EdgePointArray, CircleEdges)
    EdgePoints = OrderPoints(GetUniquePoints(EdgepointsForCircle))
    EdgePoints.remove(EdgePoints[-1])
    return EdgePoints, (EnlargedImageRowMultiplier, EnlargedImageColumnMultiplier), (imageShape[0], imageShape[1])
 

def OrderPoints(circle_points):
    ordered_list = []
    CurrPoint = circle_points[0]
    ordered_list.append(CurrPoint)
    not_done = True
    iter = 0
    while not_done:
        NextValue = (float('inf'), (0, 0))
        for point in circle_points:
            if CurrPoint == point or point in ordered_list: continue
            else:
                distance = abs(GetDistanceBetweenPoints(point, CurrPoint))
                if distance < NextValue[0]: NextValue = (distance, point)
        CurrPoint = NextValue[1]
        ordered_list.append(CurrPoint)
        iter = iter + 1
        if iter == len(circle_points): not_done = False
        
    return ordered_list


#CreateEdgeData
#Description
#This is a function creates a EdgeData structure fror each of the points calculated from the GenerateShapeEdges function

#Parameters
#FinishedList: This is the coordinates of the edge points passed in 
#imagedataClass: This holds the imageData, radius, and plane for the image

#Returns
#This returns part of edgePoints structure for each of the coordinates passed in
def CreateEdgeData(FinishedList:list, imageDataClass:ImageDataClass):
    iter = 1
    EdgeDataList = {}
    #ThreadList = []
    for edgepoint in FinishedList: #collects line information for the edgedata list
        #threadToRun = Thread(target=ThreadingFunctionForCreatingEdgedata, args=(FinishedList, EdgeDataList, edgepoint, iter, image, plane))
        #ThreadList.append(threadToRun)
        #threadToRun.start()
        if iter >= FinishedList.__len__(): NextPoint = FinishedList[0] #if we get to the last place in the array that means we've come to the point right before the beginning
        else : NextPoint = FinishedList[iter] #we first get the next point in the list
        #mutex.acquire();  mutex.release()
        iter = iter + 1
        Color =(0,0,0)
        EdgeDataList[edgepoint] = EdgeData(edgepoint, NextPoint) # creates a new edgedata
        CurrEdgePoint:EdgeData = EdgeDataList[edgepoint] # allows us to access the data class
        Linedata = CreateSolidLine(edgepoint, NextPoint) #solidifies the line we just made
        #mutex.acquire()

        for points in Linedata: EditPicture(Color, points, imageDataClass.image)
        SaveImage(imageDataClass.image, imageDataClass.plane.ImagePlaneFilePath, "View0")
        #mutex.release()
        CurrEdgePoint.__setattr__('LinePoints', Linedata)
        #for threads in ThreadList:threads.join()    
    global AdjacentPoint
    AdjacentPoint = CheckForInsideLines(imageDataClass, EdgeDataList)
    for adjacentLines in AdjacentPoint.AdjacentLine: 
        for points in AdjacentPoint.AdjacentLine[adjacentLines]: #adds on the extras lines 
            EditPicture(Color, points, imageDataClass.image)
            SaveImage(imageDataClass.image, imageDataClass.plane.ImagePlaneFilePath, "View0")

    return EdgeDataList

#ThreadingFunctionForCreatingEdgedata
#Description
#This is a function ulitizes multi-Threading to get the points around a edgePoint and then check if that point is with in the dot

#Parameters
#EdgeList: this iss the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass
#imagedataClass: This holds the imageData, radius, and plane for the image

#Returns
#This return the EDgeDataList which holds the updated Edgedata information
def CalculateLocationsOfAvaliblePixelsAroundPoint(EdgeDataList:dict, imageDataClass:ImageDataClass):
    LinePointDictionary = UnravelEdgePointLines(EdgeDataList)  #Unpacks al of the line data
    threadlist = []

    for points in EdgeDataList:
       threadToRun = Thread(target=ThreadingFunctionForMakingDotsAndCheckingCollisions, args=(points, EdgeDataList, LinePointDictionary, imageDataClass))
       threadlist.append(threadToRun)
       threadToRun.start()
       
    for threads in threadlist: threads.join() #joins the threads after they have started running
    return EdgeDataList

#UnravelEdgePointLines
#Description
#This is a function takes the points in the lines connecting the the edgepoints and gets each point in the line and adds it to a dictionary

#Parameters
#EdgeList: this is the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass

#Returns
#This returns the new dictionary with the points in the line as the key and a boolean for the value
def UnravelEdgePointLines(EdgeList:dict):
    LineDictionary = {}

    for points in EdgeList:
        edgepoint:EdgeData = EdgeList[points] 
        for linepoint in edgepoint.LinePoints:  LineDictionary[linepoint] = True

    return LineDictionary

#ThreadingFunctionForMakingDotsAndCheckingCollisions
#Description
#This function creates the area surrounding the points and checks for the collision for each of the point calucated

#Parameters
#points:This is the coordinate of the edgepoint we are testing
#imagedataClass: This holds the imageData, radius, and plane for the image
#LinePointDictionary: This is the dictionary of the points. Dict format: Key: Coordinate of linePoint (x, y). Value: Boolean
#EdgeList: this is the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass
#imagedataClass: This holds the imageData, radius, and plane for the image

#Returns
#This returns the new dictionary
def ThreadingFunctionForMakingDotsAndCheckingCollisions(points, EdgeDataList:dict, LinePointDictionary, imageDataClass:ImageDataClass):
    EdgeDataList[points].__setattr__('AllSurrondingPoints', GetFilledCircle(points, imageDataClass))#Gets all the srounding points and saves the data to the individual instances

    PointToBeCheckedForColorList = [] #this list holds the data for each of the instances

    for pointToCheck in EdgeDataList[points].AllSurrondingPoints:#loops thorugh all the points in the points surrounding the edgepoint
        EditPicture((123, 123, 124), pointToCheck, imageDataClass.image) # displays the active points on screen# we check if the points are inside the shape
        if CalculateCollision(pointToCheck, LinePointDictionary, imageDataClass) == True: PointToBeCheckedForColorList.append(pointToCheck)  #the points we want check for color
    mutex.acquire()
    SaveImage(imageDataClass.image, imageDataClass.plane.ImagePlaneFilePath, "View0")
    mutex.release()
    EdgeDataList[points].__setattr__('PointToBeCheckedForColor', PointToBeCheckedForColorList)#sets the points to be checked to the unquie instance

#CalculateCollision
#Description
#This function creates the area surrounding the points and checks for the collision for each of the point calucated

#Parameters
#pointWeCheck:This is the coordinate of the edgepoint we are testing
#imagedataClass: This holds the imageData, radius, and plane for the image
#LinePointDictionary: This is the dictionary of the points. Dict format: Key: Coordinate of linePoint (x, y). Value: Boolean

#Returns
#ReturnBool: Return true if the point is within the shape
def CalculateCollision(pointWeCheck:list, LinePointDictionary:dict, imageDataClass:ImageDataClass): #use when we grab the colors surronding each edge
    Check = 0
    XCollisonCount = 0
    YCollisonCount = 0
    ReturnBool = False

    #if we start on the line
    if LinePointDictionary.get((pointWeCheck[0], pointWeCheck[1])): ReturnBool = True

    while (ReturnBool == False and XCollisonCount < 2 and pointWeCheck[0] + Check < imageDataClass.ImageShape[0] and pointWeCheck[1] + Check < imageDataClass.ImageShape[1]):
        if LinePointDictionary.get((pointWeCheck[0], (pointWeCheck[1] + Check))): YCollisonCount = YCollisonCount + 1
        if LinePointDictionary.get(((pointWeCheck[0] + Check), round(pointWeCheck[1]))): XCollisonCount = XCollisonCount + 1 # Checks how many times our point collides with the shape
        Check = Check + 1 #if one we know the point is inside of the shape
    if XCollisonCount == 1: ReturnBool = True
    elif YCollisonCount == 1: ReturnBool = True

    return ReturnBool


#CycleThroughEdgePointsForColor
#Description
#This function loops through the pixels and stores the color inside the AverageColor list in the Edgedata

#Parameters
#pointWeCheck:This is the coordinate of the edgepoint we are testing
#imagedataClass: This holds the imageData, radius, and plane for the image
#LinePointDictionary: This is the dictionary of the points. Dict format: Key: Coordinate of linePoint (x, y). Value: Boolean

#Returns
#EdgeDataList: Returns the updated Edgedata list
def CycleThroughEdgePointsForColor(EdgeDataList, imageDataClass:ImageDataClass):
    OringalImage= cv2.imread(imageDataClass.plane.PlaneFilepath)
    AverageColorList = []

    for edges in EdgeDataList: 
        CurrEdge:EdgeData = EdgeDataList[edges]
        AverageColorList = GetAverageOfSurroundingValues(CurrEdge, OringalImage)
        CurrEdge.__setattr__('AverageColor', AverageColorList) #saves the average color for each of the instances
    
    EdgeDataList = CalculateZAxis(EdgeDataList)
    return EdgeDataList

#GetAverageOfSurroundingValues
#Description
#Calculates the average values of the indivdual points and then gets the average of those colors

#Parameters
#EdgeList: this is the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass
#oringalImage: We get the colors from the orginal image instead of the updated image

#Returns
#EdgeDataList: Returns the updated Edgedata list
def GetAverageOfSurroundingValues(EdgeDataList:EdgeData, oringalImage):
    AverageColor = []
    Colorvalues = (0, 0, 0)
    for points in EdgeDataList.PointToBeCheckedForColor: Colorvalues = [Colorvalues[0] + int(oringalImage[points][0]), Colorvalues[1] + int(oringalImage[points][1]), Colorvalues[2] + int(oringalImage[points][2])]   
    for Colors in Colorvalues: AverageColor.append(Colors / EdgeDataList.PointToBeCheckedForColor.__len__())
    return AverageColor

#CalculateZAxis
#Description
#Calculats the Z axis based on the average color of the near by points

#Parameters
#EdgeList: this is the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass
#oringalImage: We get the colors from the orginal image instead of the updated image

#Returns
#MeshStructure: the structure of the map. This dictionary has intergers as the keys and has the VErtices, Edges, and Faces 
def CalculateZAxis(EdgeDataList:dict):
    ZValuesList = []
    XList = []
    YList = []
    FinalEdgeData = []

    AdjacentEdgeZValueReference = []
    ZValueIter = 0
    for points in EdgeDataList:
        edgepoint:EdgeData = EdgeDataList[points]
        edgepoint.__setattr__('ZValue', ((round(EdgeDataList[points].AverageColor[0]) + round(EdgeDataList[points].AverageColor[1]) + round(EdgeDataList[points].AverageColor[2])) / 3))
        ZValuesList.append(EdgeDataList[points].ZValue)
        if AdjacentPoint.AdjacentLine.get(points): AdjacentEdgeZValueReference.append(ZValueIter)
        XList.append(points[0])
        YList.append(points[1])
        ZValueIter = ZValueIter + 1

    XList.append(AdjacentPoint.Coordinates[0])
    YList.append(AdjacentPoint.Coordinates[1])

    NormalisedXData = NormaliseData(XList)
    NormalisedYData = NormaliseData(YList)
    NormalisedZData= NormaliseData(ZValuesList) #Normalises the Z data so the values match the values in the orginal function

    AdjacentPoint.__setattr__('Coordinates', (NormalisedXData[-1], NormalisedYData[-1])) #grabs the normalise X and Y for the adjacent points
    iter = 0
    for ZData in NormalisedZData:
        FinalEdgeData.append((NormalisedXData[iter], NormalisedYData[iter], ZData))
        iter += 1

    iter = 0
    ZAverage = 0
    AdjacentEdgeList = []
    for points in FinalEdgeData:
        if iter in AdjacentEdgeZValueReference: 
            ZAverage += ZData
            AdjacentEdgeList.append((len(NormalisedZData)*2, iter))
        iter += 1
        ZAverage /= AdjacentEdgeZValueReference.__len__()

    TemptFinalEdgeData = FinalEdgeData.copy()
    TemptFinalEdgeData.append((AdjacentPoint.Coordinates[0], AdjacentPoint.Coordinates[1], ZAverage))
    meshMidpoint = GetMidPoint(TemptFinalEdgeData) #retrieves the midpoint from current edge data
    transposedMesh = TransposeMesh(meshMidpoint, TemptFinalEdgeData) #transposes the matrix so that that points are generated that mirror the current 3d mesh

    iter = 0
    AdjacentEdgeListTransposed = []
    for points in transposedMesh:
        if iter in AdjacentEdgeZValueReference: 
            AdjacentEdgeListTransposed.append((len(NormalisedZData)*2+1, iter+7))
        iter += 1

    TransposedAdjacentPoint = transposedMesh[-1]
    transposedMesh.remove(TransposedAdjacentPoint)

    MeshStructure = GenerateEdges(FinalEdgeData, "BlenderPoints", transposedMesh)
    MeshStructure[0].append((AdjacentPoint.Coordinates[0], AdjacentPoint.Coordinates[1], ZAverage)) #adds the adjacentPoint not transposed
    MeshStructure[0].append((TransposedAdjacentPoint)) #adds the transposed adjacent point
    MeshStructure[0].append(meshMidpoint) #adds the midpoint
    MeshStructure[1] = MeshStructure[1] + AdjacentEdgeList + AdjacentEdgeListTransposed #ands the lines connecting the adjacent points
    return MeshStructure

#NormaliseData
#Description
#Normlizes the list of point input

#Parameters
#List: The list of values that need to be normalized

#Returns
#UpdatedList: Contained the updated list with the normalised values
def NormaliseData(List:list):
    UpdatedList = []
    if not List: return False
    else: 
        for element in List:  
            if  min(List) == max(List): UpdatedList.append(0)
            else:
                norm = (element - min(List)) / (max(List) - min(List))
                UpdatedList.append(norm)
    return UpdatedList

#GenerateEdges
#Description
#Generates the meshstructure for a list of Verts passed in

#Parameters
#VertList: The list of values that need to be put into the blender meshStructre
#request: The way our meshStructre will be structure

#Returns
#UpdatedList: Contained the updated list with the normalised values
def GenerateEdges(VertList:list, request:str, VertList2:list = []):
    MeshStructure = {}
    edgeList = []
    iterator = 1

    if len(VertList) > 1:
        for verts in range(len(VertList)): #this will get the vertical edges for the mesh.
            if iterator >= len(VertList): edgeList.append((iterator-1, iterator - len(VertList)))
            else:
                edgeList.append((verts, iterator))
                iterator = iterator + 1
        iterator += 1
        for verts in range(len(VertList2)): #this will get the vertical edges for the mesh.
            if iterator >= len(VertList2)+ len(VertList): edgeList.append((iterator-1, iterator- len(VertList2)))
            else:
                edgeList.append((verts+len(VertList), iterator))
                iterator = iterator + 1 

    elif request == "BlenderPoints":
        for verts in range(VertList.__len__()): #this will get the vertical edges for the mesh.
            if iterator >= VertList.__len__(): edgeList.append((iterator-1, 0))
            else:
                edgeList.append((verts, iterator))
                iterator = iterator + 1

    if request == "BlenderPointsLineStyle":
        for verts in range(VertList.__len__()): #this will get the vertical edges for the mesh.
            edgeList.append((verts, iterator))
            iterator = iterator + 1

    elif request == "2DPoints":
        for verts in VertList: #this will get the vertical edges for the mesh.
            if iterator >=  VertList.__len__() : edgeList.append((iterator, 0))
            else:
                edgeList.append(verts)
                iterator = iterator + 1

    MeshStructure[0] = VertList + VertList2
    MeshStructure[1] = edgeList
    MeshStructure[2] = []
    return MeshStructure

def GetMidPoint(MeshStructure):
    estimateMidpoints = [] #This will hold the potential midpoints before they are averaged out
    furthestDistance = 0 #This will hold the furthest distance from one point for comparison
    for point1 in MeshStructure:
        tempPoint = list(point1) #This will hold a temporary point for whatever point we need
        for point2 in MeshStructure:
            if (point1 ==  point2): continue #if you are comparing the same point just continue on and ignore 
            else: 
                TempDistance = GetDistanceBetweenPoints3D(point1, point2) #If not then get the distance between the two points
                if (TempDistance > furthestDistance): #if that distance is greater than our current greater distance then set furthest distance to temp distance
                    furthestDistance = TempDistance
                    furthestPoint = point2
        #The next three points are getting the middle point between the first point and the furthest point
        tempPoint[0] = (point1[0] + furthestPoint[0])/2
        tempPoint[1] = (point1[1] + furthestPoint[1])/2
        tempPoint[2] = (point1[2] + furthestPoint[2])/2
        estimateMidpoints.append(tempPoint)
        furthestDistance = 0 #set fursthestDistance back to 0 and do it over again
    #Put x, y, and z elements into their own variables
    xElements = [x[0] for x in estimateMidpoints]
    yElements = [y[1] for y in estimateMidpoints]
    zElements = [z[2] for z in estimateMidpoints]
    #get the average for x, y, and z elements
    averageX = sum(xElements) / len(estimateMidpoints)
    averageY = sum(yElements) / len(estimateMidpoints)
    averageZ = sum(zElements) / len(estimateMidpoints)
    #return the midpoint constructed
    midpoint = (averageX, averageY, averageZ)
    return midpoint

def TransposeMesh(Midpoint, Mesh):
    transposedPoints = []
    for point in Mesh:
            tempPoint = list(point)
            tempPoint[0] = (Midpoint[0] + (Midpoint[0] - point[0])) #X
            tempPoint[1] = (Midpoint[1] + (Midpoint[1] - point[1])) #Y
            tempPoint[2] = (Midpoint[2] + (Midpoint[2] - point[2])) #Z
            transposedPoints.append(tempPoint)
    return transposedPoints

def GetDistanceBetweenPoints3D(point1:list, point2:list): #supporting function that just does the distance formula for 3d 
    return math.sqrt(((point2[0]-point1[0])**2) + ((point2[1]-point1[1])**2) + ((point2[2]-point1[2])**2))