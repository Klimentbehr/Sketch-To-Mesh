import cv2
from .image_processing import PlaneItem, mark_corners, EditPicture, SaveImage
from .DepthByColorHelper import AdjacentEdge, ImageDataClass, GetSlope, calucalateYIntercept, GetUniquePoints, CheckForInsideLines, CreateSolidLine, GetDistanceBetweenPoints, ColorCheck, GetFilledCircle
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
    LinePoints:list

    def __init__(self, Point ,NextPoint):
        self.Point = Point
        self.NextPoint = NextPoint
        self.slope = GetSlope(Point, NextPoint) #gets the slope of each edge
        self.Yintercept = calucalateYIntercept(Point, self.slope )

#starts from space out pixels
def GenerateShapeEdges(FullVertList:dict, radius:int, plane:PlaneItem, ColorToLookFor):
    imageDataClass = ImageDataClass(radius, plane, ColorToLookFor)

    PointArray, EnlargedImageRowMultiplier, EnlargedImageColumnMultiplier, imageShape = GetPointsFromImage(imageDataClass.image, plane, imageDataClass.ImageShape[0], imageDataClass.ImageShape[1])
    imageDataClass.__setattr__('ImageShape', imageShape)
    ImagePointArray:list = GetPointsfromPoints(FullVertList, imageDataClass.ImageShape[0], imageDataClass.ImageShape[1])
    SizedEdgeList = GetUniquePoints(PointArray + ImagePointArray)#we remove repeated values
    FinishedList = []
    for points in SizedEdgeList: FinishedList.append((round(points[0] / EnlargedImageRowMultiplier), round(points[1] / EnlargedImageColumnMultiplier)))

    imageDataClass.__setattr__('image', cv2.resize(imageDataClass.image, (imageDataClass.ImageShape[1], imageDataClass.ImageShape[0])))
    for points in FinishedList:EditPicture((0,0,0), points, imageDataClass.image)
    SaveImage(imageDataClass.image, plane.ImagePlaneFilePath, "View0")
 
    EdgeDataList = CreateEdgeData(FinishedList, imageDataClass)
    EdgeDataList = CalculateLocationsOfAvaliblePixelsAroundPoint(EdgeDataList, imageDataClass)
    outputlist = CycleThroughEdgePointsForColor(EdgeDataList, imageDataClass)
    return outputlist

def GetPointsFromImage(image, plane:PlaneItem, ImageRow, ImageColumn):
    EdgePointArray, imageShape = mark_corners(plane.PlaneFilepath)
    EdgeImageRow = imageShape[0] -1
    EdgeImageColumn = imageShape[1] -1
    EnlargedImageRowMultiplier = ImageRow / EdgeImageRow
    EnlargedImageColumnMultiplier = ImageColumn / EdgeImageColumn
    CombinedVertList = {}

    for point in EdgePointArray:
        EnlargedRow = round(point[0] * EnlargedImageRowMultiplier)
        EnlargedColummn = round(point[1] * EnlargedImageColumnMultiplier)
        CombinedVertList[EnlargedRow] = EnlargedColummn

    MaxX = max(CombinedVertList)
    GreatestX = (MaxX, CombinedVertList[MaxX])

    MinX = min(CombinedVertList)
    MiniumX = (MinX, CombinedVertList[MinX])

    MaxY = max(CombinedVertList, key = lambda k: CombinedVertList[k])
    GreatestY = (MaxY, CombinedVertList[MaxY])

    MinY = min(CombinedVertList, key = lambda k: CombinedVertList[k])
    MiniumY = (MinY, CombinedVertList[MinY])

    TopLeft = ((0, ImageColumn))
    TopRight = ((ImageRow, 0))
    BottomLeft = ((0, ImageColumn))
    BottomRight = ((ImageRow, ImageColumn))

    ShapeCorners = [TopLeft, TopRight, BottomLeft, BottomRight]
    ShapeCorner = [(0,0), (0,0), (0,0) , (0,0)] # there are four spaces for the extra points 
    ResetValue = (image.shape[1] * image.shape[0], (0,0))
    SmallestVal = (ResetValue, (0,0))
    iter = 0

    OldCornerval = ShapeCorners[0] #sets the old corner that we look at to update the iter and sets it to the current 
    for Corners in ShapeCorners:  #looks at each corner of the shape
        SmallestVal = ResetValue #resets the smallest value everytime we go to the next corner
        if not(Corners == OldCornerval) :  iter = iter + 1; OldCornerval = Corners #everytime we get a new corner we update the iter

        for points in CombinedVertList: #loops through the points in the list
            #checks if the points we are looking for are equal to any of the preivous values
            if (points, CombinedVertList[points]) == GreatestX or (points, CombinedVertList[points])  == MiniumX or (points, CombinedVertList[points])  == GreatestY or (points, CombinedVertList[points]) == MiniumY: continue

            Dst = abs(GetDistanceBetweenPoints((points, CombinedVertList[points]), Corners))
            if Dst < SmallestVal[0]: 
                SmallestVal = (Dst, (points, CombinedVertList[points]))
                ShapeCorner[iter] = SmallestVal[1]

    iterator = 0 
    iteratorArray = []
    for corners in ShapeCorner:
        if corners == (0,0): iteratorArray.append(iterator)
        iterator = iterator + 1
    for iters in iteratorArray: ShapeCorner[iters] = MiniumX

    return [MiniumY, ShapeCorner[0], MiniumX, GreatestY] , EnlargedImageRowMultiplier, EnlargedImageColumnMultiplier, (EdgeImageRow+1, EdgeImageColumn+1)

def GetPointsfromPoints(FullVertList:list, ImageRow, ImageColumn):
    CombinedVertList = {}
    for sides in FullVertList:
        for point in FullVertList[sides]:
            CombinedVertList[point[0]] = point[1]

    MaxX = max(CombinedVertList)
    GreatestX = (MaxX, CombinedVertList[MaxX])

    MinXValue = min(CombinedVertList, key=CombinedVertList.get)
    TestPointvalues1 = (MinXValue, CombinedVertList[MinXValue])

    GreatestXsmallestY = (0, 0)
    #sets the greatest value to the smallest value we can get
    GreatestCurrentValue = (0, (0,0))
    #sets the smallest value to the larger value we can get
    SmallestCurrValue = (GetDistanceBetweenPoints((0,0), (ImageRow, ImageColumn)), (ImageRow, ImageColumn))

    CurrSmallBottomvalue = SmallestCurrValue
    GreatestPointvalues = (0, (0,0))
    SmallestPointvalues = (10000, (0,0))

    for Xvalue in CombinedVertList:
        YAxis = CombinedVertList[Xvalue] #saves the y value of the list to a more readable varible
        PointDst = GetDistanceBetweenPoints((0,0), (Xvalue, YAxis))
        PointValue = Xvalue + YAxis 
        PointDstBottom = GetDistanceBetweenPoints((ImageRow, ImageColumn), (Xvalue, YAxis)) 

        if(PointValue < SmallestPointvalues[0]): SmallestPointvalues = (PointValue, (Xvalue, YAxis) )
        if(PointValue > GreatestPointvalues[0]): GreatestPointvalues = (PointValue, (Xvalue, YAxis) )

        if (PointDst > GreatestCurrentValue[0]): GreatestCurrentValue = (PointDst, (Xvalue, YAxis))
        if (PointDst < SmallestCurrValue[0]): SmallestCurrValue = (PointDst, (Xvalue, YAxis))
        
        if PointDstBottom < CurrSmallBottomvalue[0]: CurrSmallBottomvalue = (PointDstBottom, (Xvalue, YAxis))

        #we also what the hightest of the Xvalue with the Lowest of Y values and Vice versa
        if (CalculateGreatestAxisWithsmallestAxis((Xvalue, YAxis), GreatestXsmallestY, "X")) : GreatestXsmallestY = (Xvalue, YAxis)

    #GreatestY -> GreatestX ->GreatestXGreatestY ->GreatestXsmallestY -> SmallestY -> SmallestXSmallestY ->SmallestXGreatestY -> smallestX
    #we make the list in this order so they match up
    return  [GreatestX , GreatestXsmallestY, TestPointvalues1]

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
    AdjacentPoint:AdjacentEdge = CheckForInsideLines(imageDataClass, EdgeDataList)
    for adjacentLines in AdjacentPoint.AdjacentLine: 
        for points in AdjacentPoint.AdjacentLine[adjacentLines]: #adds on the extras lines 
            EditPicture(Color, points, imageDataClass.image)
            SaveImage(imageDataClass.image, imageDataClass.plane.ImagePlaneFilePath, "View0")
    return EdgeDataList

def ThreadingFunctionForCreatingEdgedata(FinishedList, EdgeDataList, edgepoint, iter, imageDataClass:ImageDataClass):
    if iter >= FinishedList.__len__(): NextPoint = FinishedList[0] #if we get to the last place in the array that means we've come to the point right before the beginning
    else : NextPoint = FinishedList[iter] #we first get the next point in the list
    mutex.acquire(); iter = iter + 1; mutex.release()
    Color =(0,0,0)
    EdgeDataList[edgepoint] = EdgeData(edgepoint, NextPoint) # creates a new edgedata
    CurrEdgePoint:EdgeData = EdgeDataList[edgepoint] # allows us to access the data class
    Linedata = CreateSolidLine(edgepoint, NextPoint) #solidifies the line we just made
    mutex.acquire()
    for points in Linedata: EditPicture(Color, points, imageDataClass.image)
    SaveImage(imageDataClass.image, imageDataClass.plane.ImagePlaneFilePath, "View0")
    mutex.release()
    CurrEdgePoint.__setattr__('LinePoints', Linedata)


def CalculateLocationsOfAvaliblePixelsAroundPoint(EdgeDataList:dict, imageDataClass:ImageDataClass):
    LinePointDictionary = UnravelEdgePointLines(EdgeDataList)  #Unpacks al of the line data
    threadlist = []

    for points in EdgeDataList:
       threadToRun = Thread(target=ThreadingFunctionForMakingDotsAndCheckingCollisions, args=(points, EdgeDataList, LinePointDictionary, imageDataClass))
       threadlist.append(threadToRun)
       threadToRun.start()
       
    for threads in threadlist: threads.join() #joins the threads after they have started running
    return EdgeDataList

def UnravelEdgePointLines(edgepointList ):
    LineDictionary = {}

    for points in edgepointList:
        edgepoint:EdgeData = edgepointList[points] 
        for linepoint in edgepoint.LinePoints:  LineDictionary[linepoint] = True

    return LineDictionary

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

def makeDot(CenterPoint:list, imageDataClass:ImageDataClass):
    CurrSurroundingVals = []

    for CurrRad in range(imageDataClass.radius):
        #if we cant move at all
        if CenterPoint[0] + CurrRad > imageDataClass.ImageShape[0] and CenterPoint[1] + CurrRad > imageDataClass.ImageShape[1] and CenterPoint[0] - CurrRad< 0 and CenterPoint[1] - CurrRad< 0: 
            return CurrSurroundingVals

        # if we can go to the left or right
        elif CenterPoint[0] + CurrRad > imageDataClass.ImageShape[0] and CenterPoint[0] - CurrRad < 0: 
            CurrSurroundingVals.append((CenterPoint[0], (CenterPoint[1] + CurrRad))) # top point
            CurrSurroundingVals.append((CenterPoint[0],(CenterPoint[1] - CurrRad))) # bottom point

        #if we cant go up or down
        elif CenterPoint[1] + CurrRad > imageDataClass.ImageShape[1] and CenterPoint[1] - CurrRad < 0: 
            CurrSurroundingVals.append(((CenterPoint[0] + CurrRad), CenterPoint[1]))# right point
            CurrSurroundingVals.append(((CenterPoint[0] - CurrRad), CenterPoint[1]))# left point

         # if we cannot go to the right
        elif CenterPoint[0] + CurrRad > imageDataClass.ImageShape[0]:
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
        elif CenterPoint[1] + CurrRad > imageDataClass.ImageShape[1]:
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
        if points[0] == imageDataClass.ImageShape[0]: continue
        elif points[1] == imageDataClass.ImageShape[1]: continue
        else: CheckedCurrSurroundingVals.append(points)

    return CheckedCurrSurroundingVals

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

def CycleThroughEdgePointsForColor(EdgeDataList, imageDataClass:ImageDataClass):
    OringalImage= cv2.imread(imageDataClass.plane.PlaneFilepath)
    AverageColorList = []

    for edges in EdgeDataList: 
        CurrEdge:EdgeData = EdgeDataList[edges]
        AverageColorList = GetAverageOfSurroundingValues(CurrEdge, OringalImage)
        CurrEdge.__setattr__('AverageColor', AverageColorList) #saves the average color for each of the instances
    
    EdgeDataList = CalculateZAxis(EdgeDataList)
    return EdgeDataList

def GetAverageOfSurroundingValues(EdgePoint:EdgeData, oringalImage):
    AverageColor = []
    Colorvalues = (0, 0, 0)
    for points in EdgePoint.PointToBeCheckedForColor: Colorvalues = [Colorvalues[0] + int(oringalImage[points][0]), Colorvalues[1] + int(oringalImage[points][1]), Colorvalues[2] + int(oringalImage[points][2])]   
    for Colors in Colorvalues: AverageColor.append(Colors / EdgePoint.PointToBeCheckedForColor.__len__())
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

    MeshStructure[0] = VertList
    MeshStructure[1] = edgeList
    MeshStructure[2] = []
    return MeshStructure