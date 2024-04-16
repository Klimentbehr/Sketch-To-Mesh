import math
import cv2
from .image_processing import PlaneItem, EditPicture, SaveImage
from dataclasses import dataclass

@dataclass
class AdjacentEdge:
    Coordinates = [] #this will hold the coordinate of the center point
    AdjacentLine = {} #this will hold the lines for the Adjacent edgeDatas   The dictionary will have a edgePoint coordinates as it's key and a list of points

    def __init__(self, Coordinates, EdgePoints):
        self.Coordinates = Coordinates
        for Nextpoints in EdgePoints:
            Linedata = SolidifyEdgePointlines(calculateLine(Coordinates, Nextpoints))
            self.AdjacentLine[Nextpoints] = Linedata


@dataclass
class ImageDataClass:
    radius = [] #this is the radius which is just the ploycount
    Color = [] # this is the color we are looking for. This is color passed in for the open Cv
    plane:PlaneItem #this is the planeItem object that has the filepaths
    image = cv2.Mat#this is the read image
    ImageShape  = [] #this is the shape of the image

    def __init__(self, radius, plane:PlaneItem, Color):
        self.radius = radius
        self.plane = plane
        self.Color = Color
        self.image = cv2.imread(plane.ImagePlaneFilePath)
        self.ImageShape = (self.image.shape[0]-1, self.image.shape[1]-1)

#CheckForInsideLines
#Description
#This function loops through the edgelist to find any lines that are not connected accounted for
#this funciton ustilises a recursive function that loops through the points in each line
#and returns the adjacent point, the edgePoint or starts the function again with the next set of points

#Parameters
#imagedataClass: This holds the imageData, radius, and plane for the image
#EdgeList: this iss the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass

#Returns:
#This returns the AdjacentEdge dataclass with the apppropriate adjcent Point data
def CheckForInsideLines(imagedataClass:ImageDataClass, EdgeList:dict):
    AdjacentPointList = [] #this holds the holds that might make up the adjcentPoint
    lastEdgePoint = (0,0) #this holds the last edgepoint recorded. This holds the adjcentPoint once we find it

    for Edgepoint in EdgeList: #we loop through the edgePoints. We are looking for a point with 3 or more lines connected to it. This point cannot be a edgePoint
        lastPointOnLine = CheckLineDst(Edgepoint, EdgeList, imagedataClass) #this will get all the points in a line till it gets to an EdgePoint or till it get 3 or more lines connected it
        if EdgeList.get(lastPointOnLine): lastEdgePoint = lastPointOnLine; continue #if we get a EdgePoint we go to the point EdgePoint in the list
        else: AdjacentPointList.append(lastPointOnLine) #if we don't get an EdgePoint we add it to the list of AdjcentPoints

    CondensedAdjacentPoints = GetPointsWithinRadius(AdjacentPointList, imagedataClass.radius) #this will combine all of the point in the Adjcent list that are close
    AdjacentPoint = GetAverageOfAllCoordinateValuesInList(CondensedAdjacentPoints) #this will combine all of the points in the list
    ConnectedEdgePoints = GetLinesAroundAdjancentPoint(AdjacentPoint, lastEdgePoint, imagedataClass, EdgeList) # this will get all of the EdgePoints connected to the EdgeList

    return GetAdjcentPointConnections(ConnectedEdgePoints, EdgeList, AdjacentPoint) # this will return the AdjacentEdge Dataclass with all of the EdgePoints connected to the AdjcentPoint

#GetAdjcentPointConnections
#Description
#This function takes in a list of points that link up with the EdgePoint and then compares the distance between those point sna d the EdgePoint.
#Once we know that distance we know the smallest value is the Edgepoint connected to the adjcentPoint
#We then add that value to a list that is used to create the adjcent data

#Parameters
#ConnectedEdgePoints: This points is the points that are a radius away from the Adjacent Edge and are on the smae line as the EdgePoint
#EdgeList: this iss the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass
#AdjacentPoint: This is the AdjcentPoint  we are checking

#Returns
#Returns the AdjacentEdge dataclass with the EdgePoint connected
def GetAdjcentPointConnections(ConnectedEdgePoints, EdgeList, AdjacentPoint):
    FinalSurrondingEdgePoints = {}
    ConfirmedConnectedEdgePoints = []
    
    for points in ConnectedEdgePoints:
        FinalSurroundingPointList = []
        for Edgepoint in EdgeList:
            FinalSurroundingPointList.append((round(abs(GetDistanceBetweenPoints(points, Edgepoint))), Edgepoint))
        FinalSurrondingEdgePoints[points] = FinalSurroundingPointList
    
    for listPointer in FinalSurrondingEdgePoints:
        ListObj:list = FinalSurrondingEdgePoints[listPointer]
        minValue = (10000000, (0,0))
        for points in ListObj:
            if points[0] < minValue[0]: minValue = points
        ConfirmedConnectedEdgePoints.append(minValue[1])

    return AdjacentEdge(AdjacentPoint, ConfirmedConnectedEdgePoints)

#GetLinesAroundAdjancentPoint
#This function creates a circle around the point passed in. 
#With that point we then get the intersections between that point a any line that hs the same color we are looking for.
#We then return that as a list of points

#Parameters
#CenterPoint: This will be the centerPoint of the circle we are drawing
#lastEdgePoint: this is the last point we know is connected to the centerPoint.
#imagedataClass: This holds the imageData, radius, and plane for the image
#EdgeList: this iss the list of edgepoints. Dict format: Key: Coordinate of EdgePoint (x, y). Value: The EdgePoint dataclass

#Returns
#SurrondingEdgePoints: This is the coordinates of the points are found on the line intersecting the cirlce we drew

def GetLinesAroundAdjancentPoint(CenterPoint, lastEdgePoint, imagedataClass:ImageDataClass, EdgeList:dict):
    SurrondingEdgePoints = [lastEdgePoint] #we start off this list with the found edgePoint
    SurrondingPoints = getCircle(CenterPoint, imagedataClass.image, imagedataClass.radius+10)
    NextPointsToCheck = CondensePointsForCircle(SurrondingPoints, imagedataClass)
    PointDistances = {}
    #saves all of the distances between the last edgePoint found and the points from the surounding points.
    #we need to do this because we don't want to find a point that we have already found
    for points in NextPointsToCheck: PointDistances[points] = abs(GetDistanceBetweenPoints(points, SurrondingEdgePoints[0]))
    NextPointsToCheck.remove(min(PointDistances, key = lambda k: PointDistances[k]))
    
    for points in NextPointsToCheck: 
        if points[1] < CenterPoint[1]: SurrondingEdgePoints.append(CheckLineDst(points, EdgeList, imagedataClass, Reverse=True))
        else: SurrondingEdgePoints.append(CheckLineDst(points, EdgeList, imagedataClass))
    
    return SurrondingEdgePoints

#CondensePointsForCircle

#Description
#This function will look for the GreenPoints in the image and then add
#them to a list to send to the GetPointsWithinRadius function. This will condense al of the greendPoints found in the Circle

#Parameters
#CirclePoints: This is the list of points being tested
#imagedataClass: This holds the imageData, radius, and plane for the imaget

#Return
#GetPointsWithinRadius(CircleGreenPoints, radius): This is the adjacent point or the EdgePoint that is found at the end of the function
def CondensePointsForCircle(CirclePoints, imagedataClass:ImageDataClass):
    CircleGreenPoints = []
    OffColor = True
    LineCounts = 0
    for point in CirclePoints:
        if ColorCheck(imagedataClass.image, point, imagedataClass.Color) and OffColor == True: #we just found a new point
            CircleGreenPoints.append(point)
            LineCounts = LineCounts + 1
        else: OffColor = True
    return GetPointsWithinRadius(CircleGreenPoints, imagedataClass.radius)

#CheckLineDst
#Description
#This function will follow a line and chack if that line hits a edgePoint or a point that has three lines coming from it.
#We use circles to tell if there is still more line left.
#This function can be reverse to take the lower value found on teh circle instead of the higher. This result in 2 separate directions the circles can take

#Parameters
#Centerpoint: This is the list of points being tested
#EdgeList: This is the list of edgePoint made from the CreateEdgePoints Function
#imagedataClass: This holds the imageData, radius, and plane for the image
#Reverse: This is used to reverse the flow of the Line check

#Return
#Adjacentpoint: This is the adjacent point or the EdgePoint that is found at the end of the function
def CheckLineDst(Centerpoint, EdgeList, imagedataClass:ImageDataClass, Reverse=False):    
    #while the point we are Currently on is still green we will check the 
    broken = False
    while True:
        EdgePointsWithDstFromCheckingPoint = {}
        Circlelist = getCircle(Centerpoint, imagedataClass.image, imagedataClass.radius) 
        NextPointsToCheck = CondensePointsForCircle(Circlelist, imagedataClass)

        for points in Circlelist: EditPicture((173, 2, 100), points, imagedataClass.image)
        for points in NextPointsToCheck: EditPicture((173, 2, 100), points, imagedataClass.image)
        SaveImage(imagedataClass.image, imagedataClass.plane.ImagePlaneFilePath, "View3")
        
        if Reverse == True:
            DeterminigValue = (100000, (0,0))
            for nextPoints in NextPointsToCheck:
                if nextPoints[1] < DeterminigValue[0]: DeterminigValue = (nextPoints[1], (nextPoints))
        elif Reverse == False:
            DeterminigValue = (0, (0,0))
            for nextPoints in NextPointsToCheck:
                if nextPoints[1] > DeterminigValue[0]: DeterminigValue = (nextPoints[1], (nextPoints))

        for edgePoints in EdgeList:
            EdgePointsWithDstFromCheckingPoint[edgePoints] = GetDistanceBetweenPoints(DeterminigValue[1], edgePoints)
            if EdgePointsWithDstFromCheckingPoint[edgePoints] < imagedataClass.radius/1.5: 
                Adjacentpoint = edgePoints
                broken = True
        if broken: break
        
        if NextPointsToCheck.__len__() >= 3:
            Adjacentpoint = GetAverageOfAllCoordinateValuesInList(NextPointsToCheck)#if this is not an edgePoint and has less that three lines coming from it we want to find the next point over
            break 
        else: Centerpoint = DeterminigValue[1]# we move on to the next point
    return Adjacentpoint

#GetPointsWithinRadius
#Description
#This function will get all the points that are close to each other and then merges those points together

#Parameters
#pointsList: This is the list of points being tested
#radius: this is the radius that we are testing

#Return
#FinalPointsList: These are the vertices that are far enough away to be consider separate points 
def GetPointsWithinRadius(pointsList:list, Radius):
    returnedPointList = []
    returnedPointList.append(pointsList[0]) 
    PointAveragingList = {}
    
    for points in pointsList:
        GoodPoint = True
        for returnPoints in returnedPointList:
            dstBetweenPoints = abs(GetDistanceBetweenPoints(points, returnPoints))
            if dstBetweenPoints < Radius: GoodPoint = False
            elif dstBetweenPoints == 0: GoodPoint = False
        if GoodPoint: returnedPointList.append(points)

    for ReturnPoints in returnedPointList:
        AvergingPointList = []
        
        for Points in pointsList:
            if abs(GetDistanceBetweenPoints(ReturnPoints, Points)) < Radius:  AvergingPointList.append(Points)
        PointAveragingList[ReturnPoints] = AvergingPointList

    FinalPointsList = []
    for AvergingPointList in PointAveragingList: FinalPointsList.append(GetAverageOfAllCoordinateValuesInList(PointAveragingList[AvergingPointList]))

    return FinalPointsList

#getCircle
#Description: This function will get a list of points in a circle around the center point.

#Parameters:
#center: this is the center of the image
#radius: this is the polyCount (this wll be updated when we get the proper image size and the change when the image is resized
#image: This is used to get the shape of the image

#Return:
#CirclePoints: these are the points making the circle
def getCircle(center:list, image, Radius = 1):
    #Orders the points in the circle
    imageShape = image.shape
    CirclePointsRT = []
    CirclePointsLT = []
    CirclePointsRB = []
    CirclePointsLB = []

    if Radius == 1: #if our radius is greater than one we need to calculate the points inbetween the Cardinal points
        right = (center[0] + Radius, center[1])
        topRight = (center[0] + Radius, center[1] + Radius)
        bottomRight = (center[0] + Radius, center[1] - Radius)
        left = ((center[0] - Radius, center[1]))
        topLeft =((center[0] - Radius, center[1] + Radius))
        bottomLeft = ((center[0] - Radius, center[1] -Radius))
        up = ((center[0], center[1] + Radius))
        down = ((center[0], center[1] -Radius))
        CirclePoints = [up, topRight, right, bottomRight, down, bottomLeft, left, topLeft]
    
    else:
        for Xvalue in range(Radius):
            Xpoint = center[0] + Xvalue
            Yvalue = round(math.sqrt(Radius**2 - (Xpoint - center[0])**2))
            if (center[0]+Xvalue < imageShape[0]) and center[1]+Yvalue < imageShape[1] :
                CircleCoord = (center[0]+Xvalue, center[1]+Yvalue); CirclePointsRT.append(CircleCoord)
            if (center[0]-Xvalue > 0) and (center[1]+Yvalue < imageShape[1]) :
                CircleCoord = (center[0]-Xvalue, center[1]+Yvalue ); CirclePointsRB.append(CircleCoord)
            if (center[1]+Xvalue < imageShape[0]) and (center[1]-Yvalue > 0) :
                CircleCoord = (center[0]+Xvalue, center[1]-Yvalue); CirclePointsLB.append(CircleCoord)
            if (center[1]-Xvalue > 0) and (center[1]-Yvalue > 0) :
                CircleCoord = (center[0]-Xvalue, center[1]-Yvalue); CirclePointsLT.append(CircleCoord)
    
    ReversedRT = CirclePointsRT[::-1]
    RightCirle:list = ReversedRT + CirclePointsRB
    ReverseLB = CirclePointsLB[::-1]
    LeftCirle:list = ReverseLB + CirclePointsLT
    TopLeftToTopRight =[]
    BottomLeftToBottomRight = []

    for Ypoints in range(RightCirle[0][1]-LeftCirle[0][1] ):
        NewPoint = (LeftCirle[0][0], LeftCirle[0][1] + Ypoints)
        TopLeftToTopRight.append(NewPoint)

    for Ypoints in range(RightCirle[-1][1] -LeftCirle[-1][1]):
        NewPoint = (LeftCirle[-1][0], (LeftCirle[-1][1] + Ypoints))
        BottomLeftToBottomRight.append(NewPoint)
    
    CirclePoints:list = TopLeftToTopRight + RightCirle + BottomLeftToBottomRight[::-1] + LeftCirle[::-1]
    return CirclePoints

#GetFilledCircle
#Description: Does the same thing as GetCircle, but fills the circle in aswell

#Parameters
#center: this is the center of the image
#radius: this is the polyCount (this wll be updated when we get the proper image size and the change when the image is resized
#image: This is used to get the shape of the image

#Return
#CirclePoints: these are the points inside and making the circle
def GetFilledCircle(center:list, imagedata:ImageDataClass):
    CirclePointsRT = []
    CirclePointsLT = []
    CirclePointsRB = []
    CirclePointsLB = []
    CircleRingList = []

    for points in range(imagedata.radius):
        if points == 0: continue
        for Xvalue in range(points):
            Xpoint = center[0] + Xvalue
            Yvalue = round(math.sqrt(imagedata.radius**2 - (Xpoint - center[0])**2))
                
            if (center[0]+Xvalue < imagedata.ImageShape[0]) and center[1]+Yvalue < imagedata.ImageShape[1] :
                CircleCoord = (center[0]+Xvalue, center[1]+Yvalue); CirclePointsRT.append(CircleCoord)
            if (center[0]-Xvalue > 0) and (center[1]+Yvalue < imagedata.ImageShape[1]) :
                CircleCoord = (center[0]-Xvalue, center[1]+Yvalue ); CirclePointsRB.append(CircleCoord)
            if (center[1]+Xvalue < imagedata.ImageShape[0]) and (center[1]-Yvalue > 0) :
                CircleCoord = (center[0]+Xvalue, center[1]-Yvalue); CirclePointsLB.append(CircleCoord)
            if (center[1]-Xvalue > 0) and (center[1]-Yvalue > 0) :
                CircleCoord = (center[0]-Xvalue, center[1]-Yvalue); CirclePointsLT.append(CircleCoord)

        ReversedRT = CirclePointsRT[::-1]
        RightCirle:list = ReversedRT + CirclePointsRB
        ReverseLB = CirclePointsLB[::-1]
        LeftCirle:list = ReverseLB + CirclePointsLT
        TopLeftToTopRight =[]
        BottomLeftToBottomRight = []

        for Ypoints in range(RightCirle[0][1]-LeftCirle[0][1] ):
            NewPoint = (LeftCirle[0][0], LeftCirle[0][1] + Ypoints)
            TopLeftToTopRight.append(NewPoint)

        for Ypoints in range(RightCirle[-1][1] -LeftCirle[-1][1]):
            NewPoint = (LeftCirle[-1][0], (LeftCirle[-1][1] + Ypoints))
            BottomLeftToBottomRight.append(NewPoint)
        
        for points in (TopLeftToTopRight + RightCirle + BottomLeftToBottomRight[::-1] + LeftCirle[::-1]): CircleRingList.append(points)

    return CircleRingList

#GetAverageOfAllCoordinateValuesInList
#Description
#Gets the Average point of the Coordinate List passed in. 

#Parameters
#CoordinateList: The List of Coordinate we want to find the Average Value of

#Returns
#The average Point found between the points passed in
def GetAverageOfAllCoordinateValuesInList(CoordinateList:list):
    Xpoints= 0
    Ypoints = 0
    for points in CoordinateList:
        Xpoints = Xpoints+ points[0]
        Ypoints = Ypoints+ points[1]
    Xpoints= round(Xpoints / CoordinateList.__len__())
    Ypoints= round(Ypoints / CoordinateList.__len__())

    return (Xpoints, Ypoints)

#GetUniquePoints
#Description
#This gets the unique values in a list

#Parameters
#Currlist: The list of points we are looking in for unique values

#Returns
#OutputList: The list of all of the unqiue values found
def GetUniquePoints(Currlist:list):
    OutputList = []
    OutputList.append(Currlist[0])

    for elements in Currlist:
        elementFound = OutputList.__contains__(elements)
        if elementFound: continue
        else: OutputList.append(elements)
    return OutputList

#ColorCheck
#Description
#This take a point on an image and compares the color value passed in to the color of the pixel specified at the point

#Parameters
#image: The Image we checking
#point: The coordinate of the pixel we want to check on the image
#Color: The color we are comparing against# The color is set to the color openCv used to outline

#Returns
#ReturnBool: This tells us if we found the value or not
def ColorCheck(image, point, Color= (0,0,255)):
    ReturnBool = False
    if point[0] >= image.shape[0] or point[1] >= image.shape[1]:  ReturnBool = False # we check if the point in within the image
    elif (int(image[point][0]), int(image[point][1]), int(image[point][2])) == Color: ReturnBool = True # we check if the color of the pixel is same as the color inputted in
    return ReturnBool

#calucalateYIntercept
#Description
#This calculates the Y intercept using the a point and the slope associated with the point

#Parameters
#Point: this is the point we want to find the YIntercept of
#Slope: This is the slope of the Point

#Returns
#Returns the YIntecept
def calucalateYIntercept(Point, slope): 
    return( Point[1] - (Point[0] * slope) )

#GetSlope
#Description
#Gets the slope of two Points

#Parameters
#points: The point we want to get the slope of

#Returns
#returnVal: The slope between point1 and point2
def GetSlope(point1:list, point2:list):
    returnVal = 0
    if point2[0] == point1[0]: returnVal = 0
    else: returnVal = (point2[1] - point1[1]) / (point2[0] - point1[0])
    return returnVal

#GetDistanceBetweenPoints
#Description
#Gets the distance between two points

#Parameters
#points: These are the points we want to get distance between

#Returns
#Returns the distance between the two points
def GetDistanceBetweenPoints(point1:list, point2:list): #supporting function that just does the distance formula
    return math.sqrt(((point2[0]-point1[0])**2) + ((point2[1]-point1[1])**2))

#GetStartingValues
#Description
#this gets the smallest point of two points and gets the difference of the two values. 
#The smaller value is alway subtracted to make sure we don't have and nergatives

#Parameters
#points: These are the points we want to get range and starting value of

#Returns
#pointDifference: This is the range between the points
#StartingPoint: This is the point the function should start at when it loops
def GetStartingValues(point1, point2):
    if point1 > point2 :
        pointDifference = point1 - point2
        StartingPoint = point2
    else : 
        pointDifference = point2 - point1
        StartingPoint = point1
    return pointDifference, StartingPoint

#CreateSolidLine
#Description
#Creates a line

#Parameters
#Points: these are point we want to use to make a line

#Returns
#Returns the points in the line we just made
def CreateSolidLine(point1, point2):
    return SolidifyEdgePointlines(calculateLine(point1, point2))

#CreateSolidLine
#Description
#Creates a line base on the line formula

#Parameters
#Points: These are the points use to create the line

#Returns
#LineData: a broken up line that needs to be solidified
def calculateLine(point1, point2):
    LineData = []
    slope = GetSlope(point1, point2) #gets the slope of each edge
    Yintercept = calucalateYIntercept(point1, slope )
    Range, StartingVal = GetStartingValues(point1[0], point2[0])
    
    for XValue in range(Range+1): #we loop through the x ranges
        XValue = StartingVal + XValue
        Yvalue = (slope * XValue) + Yintercept #we use the y = mx + c formula to find the points in the line
        LineData.append((round(XValue), round(Yvalue))) #assigns the Y value to the Xkey
    return LineData

#SolidifyEdgePointlines
#Description
#Creates a line base on the line formula

#Parameters
#Linedata: These are the line points calculated 

#Returns
#LineData: a list of points ina full line
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