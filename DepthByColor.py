import cv2
import os
import math
from .image_processing import PlaneItem
from threading import Thread, Lock
from dataclasses import dataclass

mutex = Lock()

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
    LinePoints = {}

    def __init__(self, Point ,NextPoint):
        self.Point = Point
        self.NextPoint = NextPoint
        self.slope = GetSlope(Point, NextPoint) #gets the slope of each edge
        self.Yintercept = calucalateYIntercept(Point, self.slope )

    def GetPointsToBeCheckedForColor(self, PointToBeCheckedForColor):
        self.PointToBeCheckedForColor = PointToBeCheckedForColor
#starts from space out pixels
def GenerateShapeEdges(FullVertList:dict, radius:int, plane:PlaneItem):
    CombinedVertList = {}
    image = cv2.imread(plane.ImagePlaneFilePath) 

    #this should combine all of the sides into one list 
    for sides in FullVertList:
        for point in FullVertList[sides]:
            CombinedVertList[point[0]] = point[1]

    MaxX = max(CombinedVertList)
    GreatestX = (MaxX, CombinedVertList[MaxX])
    MinX = min(CombinedVertList)
    MiniumX = (MinX, CombinedVertList[MinX])
    MaxY = max(CombinedVertList, key = lambda k: CombinedVertList[k])
    GreatestY = (MaxY, CombinedVertList[MaxY])
    MinY = min(CombinedVertList, key = lambda k: CombinedVertList[k])
    MiniumY = (MinY, CombinedVertList[MinY])

    MinXValue = min(CombinedVertList, key=CombinedVertList.get)
    TestPointvalues1 = (MinXValue, CombinedVertList[MinXValue])

    MaxXvalue = max(CombinedVertList, key=CombinedVertList.get) 
    TestPointvalues2 = (MaxXvalue, CombinedVertList[MaxXvalue])

    GreatestXsmallestY = (0, 0)
    GreatestYsmallestX = (0, 0)

    #sets the greatest value to the smallest value we can get
    GreatestCurrentValue = (0, (0,0))
    #sets the smallest value to the larger value we can get
    SmallestCurrValue = (GetDistanceBetweenPoints((0,0), (image.shape[0], image.shape[1])), (image.shape[0], image.shape[1]))

    CurrSmallBottomvalue = SmallestCurrValue
    GreatestPointvalues = (0, (0,0))
    SmallestPointvalues = (10000, (0,0))


    for Xvalue in CombinedVertList:
        YAxis = CombinedVertList[Xvalue] #saves the y value of the list to a more readable varible
        PointDst = GetDistanceBetweenPoints((0,0), (Xvalue, YAxis))
        PointValue = Xvalue + YAxis 
        PointDstBottom = GetDistanceBetweenPoints((image.shape[0], image.shape[1]), (Xvalue, YAxis)) 

        if(PointValue < SmallestPointvalues[0]): SmallestPointvalues = (PointValue, (Xvalue, YAxis) )
        if(PointValue > GreatestPointvalues[0]): GreatestPointvalues = (PointValue, (Xvalue, YAxis) )

        
        if (PointDst > GreatestCurrentValue[0]): GreatestCurrentValue = (PointDst, (Xvalue, YAxis))
        if (PointDst < SmallestCurrValue[0]): SmallestCurrValue = (PointDst, (Xvalue, YAxis))
        
        if PointDstBottom < CurrSmallBottomvalue[0]: CurrSmallBottomvalue = (PointDstBottom, (Xvalue, YAxis))

        #we also what the hightest of the Xvalue with the Lowest of Y values and Vice versa
        if (CalculateGreatestAxisWithsmallestAxis((Xvalue, YAxis), GreatestXsmallestY, "X")) : GreatestXsmallestY = (Xvalue, YAxis)
        if (CalculateGreatestAxisWithsmallestAxis((Xvalue, YAxis), GreatestXsmallestY, "Y")): GreatestYsmallestX = (Xvalue, YAxis)

    GreatestValue = GreatestCurrentValue[1]
    SmallestValue = SmallestCurrValue[1]
    BottomValue = CurrSmallBottomvalue[1]
    GreatestPointvalue = GreatestPointvalues[1]
    SmallestPointValues = SmallestPointvalues[1]

    #GreatestY -> GreatestX ->GreatestXGreatestY ->GreatestXsmallestY -> SmallestY -> SmallestXSmallestY ->SmallestXGreatestY -> smallestX
    #we make the list in this order so they match up
    EdgeList = [BottomValue, GreatestPointvalue, TestPointvalues2, GreatestX , GreatestXsmallestY, TestPointvalues1, MiniumY, SmallestValue, MiniumX, GreatestValue]
    
    #we then check for any repeated values
    FinishedList = []
    FinishedList.append(EdgeList[0]) #we add the first element in the array to get it started
    Color = (0, 0, 0)
    EditPicture(Color, EdgeList[0], image)

    for CheckingEdge in EdgeList:
        AddThis = False
        RepeatValue = False
        for elements in FinishedList:
            if CheckingEdge != elements and RepeatValue == False:  AddThis = True
            else: RepeatValue = True

        if AddThis and RepeatValue == False: 
            FinishedList.append(CheckingEdge)
            EditPicture(Color, CheckingEdge, image)
        Color[0] + 31 #updates the colors as we get through the image

    SaveImage(image, plane)
    EdgeDataList = CreateEdgeData(FinishedList, plane, image)
    EdgeDataListandImage = CalculateLocationsOfAvaliblePixelsAroundPoint(EdgeDataList, radius, plane)
    outputlist = CycleThroughEdgePointsForColor(EdgeDataListandImage, plane)
    return outputlist

def CalculateGreatestAxisWithsmallestAxis(Point1:list, Point2:list, GreaterVal:str):
    returnBool = False
    if GreaterVal == "X": GreaterVal = 0; SmallVal = 1
    else: GreaterVal = 1; SmallVal = 0
    
    if Point1[GreaterVal] > Point2[GreaterVal]:
        if Point1[SmallVal] < Point2[SmallVal]: returnBool = True
        else:
            dist1 =Point1[GreaterVal] - Point1[SmallVal]
            dist2 = Point2[GreaterVal] - Point2[SmallVal]
            if dist1 > dist2: returnBool = True
    return returnBool

def CreateEdgeData(FinishedList:list, plane:PlaneItem, image):
    iter = 1
    EdgeDataList = {}

    for edgepoint in FinishedList: #collects line information for the edgedata list
        #threadToRun = threading.Thread(target=ThreadingFunctionForCreatingEdgedata, args=(FinishedList, plane, EdgeDataList, edgepoint, iter))
        #ThreadingFunctionForCreatingEdgedata(FinishedList:list, plane:PlaneItem, EdgeDataList, edgepoint)
        if iter >= FinishedList.__len__(): NextPoint = FinishedList[0] #if we get to the last place in the array that means we've come to the point right before the beginning
        else : NextPoint = FinishedList[iter] #we first get the next point in the list
        iter = iter + 1
        
        EdgeDataList[edgepoint] = EdgeData(edgepoint, NextPoint) # creates a new edgedata
        CurrEdgePoint:EdgeData = EdgeDataList[edgepoint] # allows us to access the data class
        Linedata = SolidifyEdgePointlines(calculateLine(edgepoint, NextPoint, CurrEdgePoint.slope, CurrEdgePoint.Yintercept)) #solidifies the line we just made
        LinePointDict = {}
        for linepoints in Linedata: 
            LinePointDict[linepoints] = True
            EditPicture((0, 0, 0), (linepoints), image)
        CurrEdgePoint.__setattr__('LinePoints', LinePointDict)
    SaveImage(image, plane)
    return EdgeDataList

def calculateLine(Point, NextPoint, slope, Yintercept):
    LineData = []
    Range, StartingVal =  GetStartingValues(Point[0], NextPoint[0])
    
    for XValue in range(Range+1): #we loop through the x ranges
        XValue = StartingVal + XValue
        Yvalue = (slope * XValue) + Yintercept #we use the y = mx + c formula to find the points in the line
        LineData.append((round(XValue), round(Yvalue))) #assigns the Y value to the Xkey
    return LineData

def SolidifyEdgePointlines(Linedata:list):
    NewLinePoints = []
    iter = 1
    for point in Linedata:
        try: NextPoint = Linedata[iter]
        except: NextPoint = 0
        iter = 1 + iter

        if NextPoint == 0: break #if there is no nextPoint we are at the end of the list
        elif (point[0]+1, point[1]) == NextPoint or (point[0]-1, point[1])== NextPoint or (point[0], point[1]+1) == NextPoint or (point[0], point[1]-1) == NextPoint or (point[0]+1, point[1]+1) == NextPoint or (point[0]-1, point[1]-1) == NextPoint or (point[0]+1, point[1]-1) == NextPoint or (point[0]-1, point[1]+1) == NextPoint:
            continue
        else:
            pointDifferenceX, StartingPointX =  GetStartingValues(point[0], NextPoint[0])
            pointDifferenceY, StartingPointY =  GetStartingValues(point[1], NextPoint[1])

            for Xval in range(pointDifferenceX):
                for Yval in range(pointDifferenceY):
                    NewLinePoints.append(((StartingPointX + Xval), (StartingPointY + Yval))) #inserts the new point where it should be in the list
    
    for newpoints in NewLinePoints: Linedata.append(newpoints) #adds the new points to the dictionary
    return Linedata

def GetStartingValues(point, otherPoint):
    if point > otherPoint :
        pointDifferenceX = point - otherPoint
        StartingPointX = otherPoint
    else : 
        pointDifferenceX = otherPoint - point
        StartingPointX = point
    return pointDifferenceX, StartingPointX

def CalculateLocationsOfAvaliblePixelsAroundPoint(EdgeDataList:dict, radius:int, plane:PlaneItem):
    image = cv2.imread(plane.ImagePlaneFilePath) 
    LinePointDictionary = UnravelEdgePointLines(EdgeDataList)  #Unpacks al of the line data
    
    threadlist = []
    for points in EdgeDataList:
       threadToRun = Thread(target=ThreadingFunctionForMakingDotsAndCheckingCollisions, args=(points, EdgeDataList, image, plane, radius, [image.shape[0], image.shape[1]], LinePointDictionary))
       threadToRun.start()
       threadlist.append(threadToRun)

    for threads in threadlist: 
        threads.join() #joins the threads after they have started running
    return EdgeDataList

def UnravelEdgePointLines(edgepointList ):
    edgePointLines = {}
    for points in edgepointList:
        edgepoint:EdgeData = edgepointList[points] 
        for linepoint in edgepoint.LinePoints:  edgePointLines[linepoint] = True
    return edgePointLines

def ThreadingFunctionForMakingDotsAndCheckingCollisions(points, EdgeDataList:list, image, plane:PlaneItem, radius, imagedata, LinePointDictionary):
    #Gets all the srounding points and saves the data to the individual instances
    EdgeDataList[points].__setattr__('AllSurrondingPoints', makeDot(EdgeDataList[points].Point, radius, imagedata))
    PointToBeCheckedForColorList = [] #this list holds the data for each of the instances

    for pointToCheck in EdgeDataList[points].AllSurrondingPoints:#loops thorugh all the points in the points surrounding the edgepoint
        EditPicture((123, 123, 124), pointToCheck, image) # displays the active points on screen
        
        if CalculateCollision(pointToCheck, imagedata, LinePointDictionary) == True: #we check if the points are inside the shape
            PointToBeCheckedForColorList.append(pointToCheck)  #the points we want check for color
            EditPicture((100, 100, 255), pointToCheck, image)  # displays the active points on screen

    
    EdgeDataList[points].__setattr__('PointToBeCheckedForColor', PointToBeCheckedForColorList)#sets the points to be checked to the unquie instance
    mutex.acquire()
    SaveImage(image, plane)#saves the new image
    mutex.release()

def makeDot(CenterPoint:list, radius, imagedata):
    CurrSurroundingVals = []

    for CurrRad in range(radius):
        #if we cant move at all
        if CenterPoint[0] + CurrRad > imagedata[0] and CenterPoint[1] + CurrRad > imagedata[1] and CenterPoint[0] - CurrRad< 0 and CenterPoint[1] - CurrRad< 0: 
            return CurrSurroundingVals

        # if we can go to the left or right
        elif CenterPoint[0] + CurrRad > imagedata[0] and CenterPoint[0] - CurrRad < 0: 
            CurrSurroundingVals.append((CenterPoint[0], (CenterPoint[1] + CurrRad))) # top point
            CurrSurroundingVals.append((CenterPoint[0],(CenterPoint[1] - CurrRad))) # bottom point

        #if we cant go up or down
        elif CenterPoint[1] + CurrRad > imagedata[1] and CenterPoint[1] - CurrRad < 0: 
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), CenterPoint[1]))# right point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), CenterPoint[1]))# left point

         # if we cannot go to the right
        elif CenterPoint[0] + CurrRad > imagedata[0]:
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), CenterPoint[1])) # left point
            CurrSurroundingVals.append((CenterPoint[0], (CenterPoint[1] + CurrRad))) # top point
            CurrSurroundingVals.append((CenterPoint[0],(CenterPoint[1] - CurrRad))) # bottom point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), (CenterPoint[1] + CurrRad))) # left point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad) ,(CenterPoint[1] - CurrRad)))# bottom point

        #if we cannot go to the left
        elif CenterPoint[0] - CurrRad < 0:
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), CenterPoint[1]))# right point
            CurrSurroundingVals.append((CenterPoint[0], (CenterPoint[1] + CurrRad))) # top point
            CurrSurroundingVals.append((CenterPoint[0],(CenterPoint[1] - CurrRad))) # bottom point
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), (CenterPoint[1] + CurrRad))) # right point
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), (CenterPoint[1] - CurrRad))) # top point

        
        #if we cant go down
        elif CenterPoint[1] - CurrRad < 0:
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), CenterPoint[1])) # right point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), CenterPoint[1])) # left point
            CurrSurroundingVals.append((CenterPoint[0], (CenterPoint[1] + CurrRad))) # top point
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), (CenterPoint[1] + CurrRad))) # right point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), (CenterPoint[1] + CurrRad))) # left point


        #if we cant go up
        elif CenterPoint[1] + CurrRad > imagedata[1]:
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), CenterPoint[1])) # right point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), CenterPoint[1])) # left point
            CurrSurroundingVals.append((CenterPoint[0],(CenterPoint[1] - CurrRad))) # bottom point
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), (CenterPoint[1] - CurrRad)))# top point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad) ,(CenterPoint[1] - CurrRad))) # bottom point


        #if we have no restrictions
        else:
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), CenterPoint[1])) # right point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), CenterPoint[1])) # left point
            CurrSurroundingVals.append((CenterPoint[0], (CenterPoint[1] + CurrRad))) # top point
            CurrSurroundingVals.append((CenterPoint[0],(CenterPoint[1] - CurrRad))) # bottom point

            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), (CenterPoint[1] + CurrRad))) # right point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), (CenterPoint[1] + CurrRad))) # left point
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), (CenterPoint[1] - CurrRad))) # top point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad) ,(CenterPoint[1] - CurrRad))) # bottom point

            for vals in range(CurrRad):
                if vals == 0: continue #skips the 0
                else:
                    CurrSurroundingVals.append(((CenterPoint[0] - CurrRad + vals), (CenterPoint[1] + vals)))#leftpoint to TopPoint
                    CurrSurroundingVals.append(((CenterPoint[0] - CurrRad + vals), (CenterPoint[1] - vals)))#leftpoint to BottomPoint
                    CurrSurroundingVals.append(((CenterPoint[0] + CurrRad - vals), (CenterPoint[1] + vals)))#rightpoint to TopPoint
                    CurrSurroundingVals.append(((CenterPoint[0] + CurrRad - vals), (CenterPoint[1] - vals)))#rightpoint to BottomPoint

    CheckedCurrSurroundingVals = []
    for points in CurrSurroundingVals:
        if points[0] == imagedata[0]: continue
        elif points[1] == imagedata[1]: continue
        else: CheckedCurrSurroundingVals.append(points)

    return CheckedCurrSurroundingVals

def CalculateCollision(pointWeCheck:list, imagedata:list, LinePointDictionary:dict): #use when we grab the colors surronding each edge
    XCheck = pointWeCheck[0]
    CollisonCount = 0
    ReturnBool = False
    
    while (CollisonCount < 2 and XCheck < imagedata[0] ):
        if LinePointDictionary.get((XCheck, round(pointWeCheck[1]))): 
            CollisonCount = CollisonCount + 1 # Checks how many times our point collides with the shape           
        XCheck = XCheck + 1 #if one we know the point is inside of the shape
    if CollisonCount == 1: ReturnBool = True

    return ReturnBool

def CycleThroughEdgePointsForColor(EdgeDataList, plane:PlaneItem):
    outputlist = []
    ImageToBeResized = cv2.imread(plane.PlaneFilepath)
    AverageColorList = []
    image = cv2.resize(ImageToBeResized, (800, 600)) #we resize the orginal image to get the colors that are present in the orginal

    for edges in EdgeDataList: 
        CurrEdge:EdgeData = EdgeDataList[edges]
        AverageColorList = GetAverageOfSurroundingValues(CurrEdge, image)
        CurrEdge.__setattr__('AverageColor', AverageColorList) #saves the average color for each of the instances
    
    EdgeDataList = CalculateZAxis(EdgeDataList)
    return EdgeDataList

def GetAverageOfSurroundingValues(EdgePoint:EdgeData, image):
    AverageColor = []
    Colorvalues = (0, 0, 0)
    for points in EdgePoint.PointToBeCheckedForColor: 
        Colorvalues = [Colorvalues[0] + int(image[points][0]), Colorvalues[1] + int(image[points][1]), Colorvalues[2] + int(image[points][2])]   

    for Colors in Colorvalues: 
        AverageColor.append(Colors / EdgePoint.PointToBeCheckedForColor.__len__())
    return AverageColor

def CalculateZAxis(EdgeDataList:dict):
    ZValuesList = []
    XList = []
    YList = []
    FinalEdgeData = []

    for points in EdgeDataList:
        edgepoint:EdgeData = EdgeDataList[points]
        edgepoint.__setattr__('ZValue', ((round(EdgeDataList[points].AverageColor[0]) + round(EdgeDataList[points].AverageColor[1]) + round(EdgeDataList[points].AverageColor[2])) / 3))
        ZValuesList.append(EdgeDataList[points].ZValue)
        XList.append(points[0])
        YList.append(points[1])

    NormalisedXData = NormaliseData(XList)
    NormalisedYData = NormaliseData(YList)
    NormalisedZData= NormaliseData(ZValuesList) #Normalises the Z data so the values match the values in the orginal function

    iter = 0
    for ZData in NormalisedZData:
        FinalEdgeData.append((NormalisedXData[iter], NormalisedYData[iter], ZData))
        iter = iter + 1
    MeshStructure = GenerateEdges(FinalEdgeData, "BlenderPoints")
    return MeshStructure

def calucalateYIntercept(Point, slope):
    return( Point[1] - (Point[0] * slope) )

def GetSlope(point1:list, point2:list):
    returnVal = 0
    if point2[0] == point1[0]: returnVal = 0
    else: returnVal = (point2[1] - point1[1]) / (point2[0] - point1[0])
    return returnVal

def NormaliseData(List:list):
    NewList = []
    if not List: return False
    else: 
        for element in List:  
            if  min(List) == max(List):
                NewList.append(0)
            else:
                norm = (element - min(List)) / (max(List) - min(List))
                NewList.append(norm)
    return NewList

def GenerateEdges(VertList:list, request:str):
    MeshStructure = {}
    edgeList = []
    iterator = 1
    if request == "BlenderPoints":
        for verts in range(VertList.__len__()): #this will get the vertical edges for the mesh.
            if iterator >= VertList.__len__(): edgeList.append((iterator-1, 0))
            else:
                edgeList.append((verts, iterator))
                iterator = iterator + 1

    elif request == "2DPoints":
        for verts in VertList: #this will get the vertical edges for the mesh.
            if iterator >=  VertList.__len__() : edgeList.append((iterator, 0))
            else:
                edgeList.append(verts)
                iterator = iterator + 1

    MeshStructure[0] = VertList
    MeshStructure[1] = edgeList
    MeshStructure[2] = []
    return MeshStructure

def EditPicture(Color:list, Point:list, image):
    image[Point][0] = Color[0]
    image[Point][1] = Color[1]
    image[Point][2] = Color[2]

def SaveImage(image, plane:PlaneItem):
    os.curdir
    Extension =  plane.PlaneFilepath[plane.PlaneFilepath.rfind("."): ] 
    os.chdir("ImageFolder") #changes the directory to the folder where we are going to save the file
    cv2.imwrite("View0" + Extension, image ) #saves the image
    os.chdir("..\\") #goes back one directory   
    
def GetDistanceBetweenPoints(point1:list, point2:list): #supporting function that just does the distance formula
    return math.sqrt(((point2[1]-point1[0])**2) + ((point2[1]-point1[1])**2))