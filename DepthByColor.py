import cv2
import math
import bpy
import bmesh
import mathutils
from .image_processing import PlaneItem, mark_corners, EditPicture, SaveImage
from .DepthByColorHelper import AdjacentEdge, ImageDataClass, GetSlope, calucalateYIntercept, GetUniquePoints, CheckForInsideLines, CreateSolidLine, GetDistanceBetweenPoints, ColorCheck, GetFilledCircle, GetAverageOfAllCoordinateValuesInList, GetAverageDstBetweenPoints, getCircle, GetClosetPointsToValue, GetUniquePoints
from threading import Thread, Lock
from dataclasses import dataclass

mutex = Lock()
meshMidpoint = None
visiblePoints = 0

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

@dataclass
class CameraPoint:
    m_Coordinate:list#contains teh coordinate of the CameraPoint
    m_CenterPoint:int # contains the point the camera will rotate around
    m_MaxDst:int # contians the radius and Max dst away any cameraPoint can be
    m_Axis:list # [X, Y, Z] the axis contians the rotations for each Axis

    def __init__(self, MaxDst, CenterPoint, Axis):
        self.m_CenterPoint = CenterPoint
        self.m_MaxDst = MaxDst
        self.m_Axis = Axis
        self.m_Coordinate = [CenterPoint[0] + MaxDst, CenterPoint[1], CenterPoint[2]] #this will be the first point with (0,0,0) and will be (MaxDst, 0, 0)
        self.m_Coordinate = ChangeCoordinate(self.m_MaxDst, self.m_Axis) # this will change the coordinate based on the axis

    def ChangeAxis(self, NewAxis):
        self.m_Axis = NewAxis
        self.m_Coordinate = ChangeCoordinate(self.m_Coordinate, self.m_Axis)
    
    def CheckDstBetweenCoordinate(self, Coordinate):
        distanceBetweenPoints = GetDistanceBetweenPoints3D(self.m_Coordinate, Coordinate)
        return distanceBetweenPoints


def ChangeCoordinate(MaxDst, Axis):
    New_CoordXandY = CalculateRotation(MaxDst, Axis[0]) 
    New_CoordYandZ = CalculateRotation(MaxDst, Axis[2])
    return  [New_CoordXandY[0], New_CoordYandZ[0], New_CoordYandZ[1]]

#CalculateRotation
#Description
#This function will calucalute the rotation of a point on a circle. It uses the 
def CalculateRotation(MaxDst, angle): 
    x=MaxDst * math.cos(angle)
    y=MaxDst * math.cos(angle)
    return [x, y]

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
    AveragePoint = ((round(AveragePoint[0]), round(AveragePoint[1])))

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
    OringalImage = cv2.resize(OringalImage, (200, 200))
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
    ZValueIter = 0
    AdjacentEdgeZValueReference = []
    for points in EdgeDataList:
        edgepoint:EdgeData = EdgeDataList[points]
        edgepoint.__setattr__('ZValue', ((round(EdgeDataList[points].AverageColor[0]) + round(EdgeDataList[points].AverageColor[1]) + round(EdgeDataList[points].AverageColor[2])) / 3))

        if AdjacentPoint.AdjacentLine.get(points): AdjacentEdgeZValueReference.append(ZValueIter)
        ZValueIter += 1

    PointsToNormalise, BlenderEdgeData = GenerateMeshEdgeData(EdgeDataList, AdjacentPoint, AdjacentEdgeZValueReference)
    return CreateBlenderData(PointsToNormalise, BlenderEdgeData)



def CreateBlenderData(PointsToNormalize, BlenderEdgeData):
    XList = []; YList = []; ZList = []

    for points in PointsToNormalize: XList.append(points[0]); YList.append(points[1]); ZList.append(points[2])
    NormalisedXData = NormaliseData(XList); NormalisedYData = NormaliseData(YList); NormalisedZData= NormaliseData(ZList)

    FinalVertexPoints =[]
    for pointvalue in range(len(XList)): FinalVertexPoints.append((NormalisedXData[pointvalue], NormalisedYData[pointvalue], NormalisedZData[pointvalue]))
   
    MeshStructure = {}
    MeshStructure[0] = FinalVertexPoints
    MeshStructure[1] = BlenderEdgeData
    MeshStructure[2] = []
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

def GenerateMeshEdgeData(EdgeList:dict, AdjacentPoint:AdjacentEdge, AdjacentEdgeZValueReference:list):
    VertexList = []
    #We need the orginal coordinates to reference the EdgeDataList
    for points in EdgeList:
        point:EdgeData = EdgeList[points]
        Currpoint = (point.Point[0], point.Point[1], round(point.ZValue))
        VertexList.append(Currpoint)

    iter = 0
    AdjacentZValue = 0
    EdgeDataForBlender = []
    TAdjacentEdgePoints =[]
    for Edges in EdgeList:
        point:EdgeData = EdgeList[Edges]
        #we get the point we are currently on and the the next point in the edgelist
        EdgeDataForBlender.append(((point.Point[0], point.Point[1], round(point.ZValue)) , (point.NextPoint[0], point.NextPoint[1], round(EdgeList[point.NextPoint].ZValue))))
        #we check if any of the values are connected to the adjacent point
        if iter in AdjacentEdgeZValueReference: AdjacentZValue += point.ZValue
        else: TAdjacentEdgePoints.append((point.Point[0], point.Point[1], round(point.ZValue)) )
        iter += 1

    AdjacentPoint3D = (AdjacentPoint.Coordinates[0], AdjacentPoint.Coordinates[1], AdjacentZValue) #We store the Adjacent in a varible for easier use
    AdjacentZValue = round(AdjacentZValue/ len(AdjacentEdgeZValueReference))# we get the Z value of the adjacent point by getting the average of the Z values that are connected to the point 
    TAdjacentPoint = ((AdjacentPoint.Coordinates[0], AdjacentPoint.Coordinates[1], (-1* AdjacentZValue)))

    AdjacentDataForBlender = []
    iter = 0
    for adjacentEdges in AdjacentPoint.AdjacentLine:
            EdgePoint3D = (EdgeList[adjacentEdges].Point[0], EdgeList[adjacentEdges].Point[1], round(EdgeList[adjacentEdges].ZValue))
            AdjacentDataForBlender.append((AdjacentPoint3D, EdgePoint3D))
            AdjacentDataForBlender.append((TAdjacentPoint, TAdjacentEdgePoints[iter]))
            iter += 1

    # need our placements to know where to start our faces
    AdjacentPointLocation = len(VertexList)
    TAdjacentPointLocation= len(VertexList)+1

    #adds our adjacent points to the edgelist
    VertexList.append(AdjacentPoint3D); VertexList.append(TAdjacentPoint)
    CoordinateEdgedata = EdgeDataForBlender + AdjacentDataForBlender
    EdgeDataList = map_coordinates_to_indices(CoordinateEdgedata)
    AdjacentfaceList = RepackageFaceList(CreatFaceData(AdjacentPointLocation, EdgeDataList))
    TAdjacentfaceList = RepackageFaceList(CreatFaceData(TAdjacentPointLocation, EdgeDataList))

    for tuples in AdjacentfaceList: EdgeDataList.append(tuples)
    for tuples in TAdjacentfaceList: EdgeDataList.append(tuples)

    return VertexList, EdgeDataList

def RepackageFaceList(faceList):
    FaceSetTupleList = []
    FaceSetList = []

    for sets in faceList: 
        for pairs in sets: FaceSetTupleList.append(pairs[0]) 
        FaceSetList.append(tuple(FaceSetTupleList))
        FaceSetTupleList = []
    return FaceSetList

def CreatFaceData(AdjacentPointLocation, EdgeDataList):
    NotDone = True
    Currpoint = AdjacentPointLocation
    KnownPairs = []
    FacePairs = []

    CurrentFaces = 0
    StartingBool = True
    LastPoint = False
    Faces = {}

    while NotDone:
        for Pairs in EdgeDataList:
            if Pairs[0] == Currpoint and Pairs not in KnownPairs:
                FacePairs.append(Pairs)
                Currpoint = Pairs[1]
                CurrEdge = Pairs
                break
            elif Pairs == EdgeDataList[-1]: LastPoint = True


        if ((AdjacentPointLocation, CurrEdge[0]) in EdgeDataList and (AdjacentPointLocation, CurrEdge[0]) not in KnownPairs) or LastPoint: #we know we got back to the orginalPoint
            if LastPoint:
                FacePairs.append((KnownPairs[0][1], KnownPairs[0][0]))
                Faces[CurrentFaces] = FacePairs
                CurrentFaces += 1
            elif StartingBool == False:
                FacePairs.pop(-1) # removes the last pair before saving hte list of known points
                KnownPairs = KnownPairs + FacePairs
                FacePairs.append((CurrEdge[0], AdjacentPointLocation))
                Faces[CurrentFaces] = FacePairs
                FacePairs = []
                Currpoint = AdjacentPointLocation
                CurrentFaces += 1
                StartingBool = True
            else: StartingBool = False
        if CurrentFaces >= len(AdjacentPoint.AdjacentLine): NotDone = False

    FaceList = []
    for face in Faces: FaceList.append(Faces[face])
    return FaceList

def GetMidPoint(MeshStructure):
    estimateMidpoints = [] #This will hold the potential midpoints before they are averaged out
    furthestDistance = 0 #This will hold the furthest distance from one point for comparison
    tempPoint = [1, 2, 3] #This will hold a temporary point for whatever point we need
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

# Function to map a list of coordinate pairs to integer indices
def map_coordinates_to_indices(user_sequence):
    point_index = {}  # This dictionary will map points to integers
    index = 0
    indexed_pairs = []

    for pair in user_sequence:
        # Unpack the pair for clarity
        start, end = pair
        
        # Check if the start coordinate is new, if so, add to the point_index
        if start not in point_index:
            point_index[start] = index
            index += 1
            
        # Check if the end coordinate is new, if so, add to the point_index
        if end not in point_index:
            point_index[end] = index
            index += 1

        # Append the index pair to the output list
        indexed_pairs.append((point_index[start], point_index[end]))

    return indexed_pairs

def RecalulateVertices(VisibleVerts:list, VertsForSecondPerpesctive, VisbleEdgeDataList:list, Vertlist):
    #we need the connections between each vert and we need the verts that are not seen to be removed
    #we need to consider how many point are connected to each vert
    #We also need to consider how close the points are to the verts in the SecondPerpective
    
    VertToEdgeForOringinal = VertAsIndicator(Vertlist, VisibleVerts)
    VertToEdgeForSecondImage = VertAsIndicator(Vertlist, VertsForSecondPerpesctive)
    SimilarityList = []
    SimilartyDict = {}
    iter = 0
    for vertconnections in VertToEdgeForOringinal:
        for vertConnectForSecond in VertToEdgeForSecondImage:
            if len(VertToEdgeForOringinal[vertconnections]) == len(VertToEdgeForSecondImage[vertConnectForSecond]): SimilarityList.append(vertConnectForSecond)
        SimilartyDict[vertconnections] = SimilarityList
        iter += 1

    #Now we know which point have the same amount of vertices we can see which ones are teh closest
    NewVertList = []
    for Connections in SimilartyDict:
            ClosestPoint = GetClosestPoint3D(Connections, SimilarityList[Connections])
            NewVertList.append(GetAverageOfAllCoordinateValuesInList([ClosestPoint, Connections]))

    iter =0
    for iterator in range(len(NewVertList)):
        if iterator in VertToEdgeForOringinal:
            Vertlist[iterator] = NewVertList[iter]
            iter += 1

    return Vertlist
        
def GetClosestPoint3D(Target, PointList):
    Distances = {}
    for points in PointList: Distances[GetDistanceBetweenPoints3D(Target, points)] = points
    return Distances[min(Distances)]

        
def VertAsIndicator(FullPointlist, ComparingList, VisbleEdgeDataList):
    VertToEdge = []
    VertConnections = {}
    iter = 0
    #we will get the indicator for the point within the edgelist
    for Vertpoints in FullPointlist:
        for points in ComparingList:
            if points == Vertpoints:
                VertToEdge.append(iter) #this is the index of the vert in the edgeList
                break
        iter +=1

    #We get the verts and all of there connections in a dictionary
    #with this we know how many verts are connected to each point
    for edges in VisbleEdgeDataList:
        VertConnectionList=[]
        for vert in VertToEdge:
            if edges[0] == vert:
                VertConnectionList.append(edges[1])
        VertConnections[edges[0]] = VertConnectionList# we assign the connections to the vert
    return VertConnections

  
def ResetNormals(collectionName): #This function will reset the normals of the mesh once it has been generated
    collection = bpy.data.collections.get(collectionName) #We get the collection once the collection has been made

    if collection: #for all of the objects in the collection we will be reseting their normals
        for obj in collection.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.object.mode_set(mode='OBJECT')
    # else: self.report({'ERROR'}, "Invalid Image") 

def CountVerticesFromCamera(): #This function will count the amount of verticies visible relative to a camera
    bm = bmesh.new() #bmesh grabs the data from a mesh in blender
    camera = bpy.data.objects['Camera'] #The camera we will be counting our points from
    scene = bpy.context.scene #Scene is needed to grab x and y coordinates
    obj = bpy.context.object #grabbing the active object
    desgraph = bpy.context.evaluated_depsgraph_get() #this is specifically used on the projection matrix
    global visiblePoints

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj #setting the active object

    bm.from_mesh(obj.data) #grabbing the mesh's data

    #These two objects are the perspective matrixes of the camera which we will compare with the bmesh to see if they are inside the camera's view
    cameraMatrix = camera.matrix_world.normalized().inverted()
    cameraProjectionMatrix = camera.calc_matrix_camera(desgraph, x=scene.render.resolution_x, y=scene.render.resolution_y, scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)

    for vertex in bm.verts:
        if IsVertexVisible(vertex, cameraMatrix, cameraProjectionMatrix): 
            visiblePoints = visiblePoints + 1

def IsVertexVisible(vertex, cameraMatrix, cameraProjectionMatrix): #Assistant function to the function above
    cameraLocal = cameraMatrix @ vertex.co
    cameraWorld = cameraProjectionMatrix @ cameraLocal.to_4d()

    if cameraWorld.w > 0:
        cameraWorld /= cameraWorld.w
        if -1 <= cameraWorld.x <= 1 and -1 <= cameraWorld.y <= 1:
            return True
    return False
