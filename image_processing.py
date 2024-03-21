import cv2
import os
import numpy as np
import bpy
from dataclasses import dataclass

@dataclass
class PlaneItem:
    PlaneFilepath = bpy.props.StringProperty(name="File Path",subtype='FILE_PATH')
    PlaneRotation = bpy.props.IntProperty(name="Rotation", default=0)
    ImagePlaneName: str
    ImagePlaneFilePath: str
    
    def __init__(self, filepath ,rotation):
        self.PlaneFilepath = filepath
        self.PlaneRotation = rotation


# this will be called once the images are ready
def prepare_image(image_path):
    
    image = cv2.imread(image_path)

    # temporary file size. adjusting files to the same scale can be beneficial for feature detectors
    resized_image = cv2.resize(image, (800, 600))  
    
    # grayscale reduces computational load
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) 

    # noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # canny edge detection emphasizes edges in the image
    # we will most likely be using one of the two as feature detectors: ORB, AKAZE.
    # both feature detection algorithms have positive results from this as they often rely on edge information to find key points.
    edges = cv2.Canny(blurred_image, 50, 150)

    # defines region of interest inside of the image.
    # this will most likely not be necessary.
    
    # mask = np.zeros_like(edges)
    # roi_vertices = np.array([[(50, 600), (750, 600), (400, 100)]], dtype=np.int32)
    # cv2.fillPoly(mask, roi_vertices, 255)
    # masked_edges = cv2.bitwise_and(edges, mask)
    
    try:
        # its just going to try to connect and list db collection names
        output_path = os.path.join('C:/Users/RAFAEL MUITO ZIKA/Desktop/Test', 'prepared_image.png')
        cv2.imwrite(output_path, edges)
        print(f"Image prep done.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
# wth is this? why is this here and not in a proper file? why does this method even exists? sigh.
def outline_image(image_path, Extension, ImgName, Filedirectory):
    """Read an image from a path, outline it, calculate the center of mass for the outlines, and draw a blue dot there."""
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found or unable to load.")
        return
    
   # temporary file size. adjusting files to the same scale can be beneficial for feature detectors
    resized_image = cv2.resize(image, (800, 600))  
    
    # grayscale reduces computational load
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) 

    # noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # canny edge detection emphasizes edges in the image
    # we will most likely be using one of the two as feature detectors: ORB, AKAZE.
    # both feature detection algorithms have positive results from this as they often rely on edge information to find key points.
    edges = cv2.Canny(blurred_image, 50, 150)

    # defines region of interest inside of the image.
    # this will most likely not be necessary.
    
    # mask = np.zeros_like(edges)
    # roi_vertices = np.array([[(50, 600), (750, 600), (400, 100)]], dtype=np.int32)
    # cv2.fillPoly(mask, roi_vertices, 255)
    # masked_edges = cv2.bitwise_and(edges, mask)
    
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if contours:
        # Draw contours on the original image
        cv2.drawContours(resized_image, contours, -1, (0, 255, 0), 2)
        # Calculate the combined center of mass for all contours
        totalX, totalY, totalArea = 0, 0, 0

        for contour in contours:
            M = cv2.moments(contour)

            if M["m00"] != 0:
                totalX += int(M["m10"])
                totalY += int(M["m01"])
                totalArea += M["m00"]

        if totalArea != 0:
            cX = int(totalX / totalArea)
            cY = int(totalY / totalArea)
            # Draw a blue dot at the combined center of mass
            cv2.circle(resized_image, (cX, cY), 5, (255, 0, 0), -1)
        else:
            print("Error: Combined center of mass could not be calculated.")
    else:
        print("Error: No contours found.")

    try:
        os.chdir(Filedirectory) #changes the directory to the folder where we are going to save the file
        cv2.imwrite(ImgName + Extension, resized_image) #saves the image
        os.chdir("..\\") #goes back one directory   
        print(f"Image prep done.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def match_features(descriptors1, descriptors2, method='ORB'):
    # using ORB and AKAZE for testing
    if method == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:  
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # knn Matching descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # RATIO TEST
    # filters out weak matches by comparing the distance of the closes neighbor to that of the second closest neighbor.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Ratio test
            good_matches.append(m)
    
    return good_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

def detect_and_describe_akaze(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise IOError("Could not open the image from path: {}".format(image_path))
    
    akaze = cv2.AKAZE_create()
    keypoints = akaze.detect(image, None)
    
    # compute descriptors for the detected keypoints
    keypoints, descriptors = akaze.compute(image, keypoints)
    
    return keypoints, descriptors

def test_feature_detection():
    img_path1 = 'C:/Users/Rafael/Desktop/Exampel/side.png'
    img_path2 = 'C:/Users/Rafael/Desktop/Exampel/sidee.png'
    img_path3 = 'C:/Users/Rafael/Desktop/Exampel/front.png' 

    # detect
    keypoints1, descriptors1 = detect_and_describe_akaze(img_path1)
    keypoints2, descriptors2 = detect_and_describe_akaze(img_path2)
    keypoints3, descriptors3 = detect_and_describe_akaze(img_path3)

    # match
    matches12 = match_features(descriptors1, descriptors2, method='AKAZE')
    matches23 = match_features(descriptors2, descriptors3, method='AKAZE')
    matches13 = match_features(descriptors1, descriptors3, method='AKAZE')

    # visualize
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)
    image3 = cv2.imread(img_path3)

    matched_image12 = draw_matches(image1, keypoints1, image2, keypoints2, matches12)
    matched_image23 = draw_matches(image2, keypoints2, image3, keypoints3, matches23)
    matched_image13 = draw_matches(image1, keypoints1, image3, keypoints3, matches13)

    # display
    cv2.imshow('Matches between Image 1 and Image 2', matched_image12)
    cv2.imshow('Matches between Image 2 and Image 3', matched_image23)
    cv2.imshow('Matches between Image 1 and Image 3', matched_image13)

    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

def PlaceImage(self, GlobalPlaneDataArray:list[PlaneItem] ):
     #this will keep count of the views were have captured
        Itervalue = 0
        #this will be a folder in the sketch-to-Mesh project. This will hold the Image processed
        ImageDiretoryForNewImage = "ImageFolder"
        #this will eventually need to move to somewhere more accessable
        #this is only here for now
    
        for plane_data in GlobalPlaneDataArray : 
            if plane_data :
                #this is used for the new image. We want to save the new image as the same type of file as the first
                Extension =  plane_data.PlaneFilepath[plane_data.PlaneFilepath.rfind("."): ] 
                plane_data.ImagePlaneName = "View" + str(Itervalue) #this is the file name for the image we are creating
                # allows us to access the plane after creation
                outline_image(plane_data.PlaneFilepath, Extension, plane_data.ImagePlaneName, ImageDiretoryForNewImage)
                #this creates a new file path to the image we just saved
                plane_data.ImagePlaneFilePath = os.path.abspath(ImageDiretoryForNewImage + "\\" + plane_data.ImagePlaneName + Extension) 
                
                if plane_data.ImagePlaneFilePath:
                    filename = os.path.basename(plane_data.ImagePlaneFilePath)
                    FileDirectory = r"" + plane_data.ImagePlaneFilePath[: plane_data.ImagePlaneFilePath.rfind("\\")] + "\\"
                    #bpy.ops.import_image.to_plane(files=[{"name":filename, "name":filename}], directory=FileDirectory, relative=False)
                    bpy.ops.import_image.to_plane(files=[{"name":filename, "name":filename}], directory=FileDirectory, relative=False)
                    #we set the rotation and location of each plane
                    bpy.data.objects[plane_data.ImagePlaneName].select_set(True)
                    match Itervalue :
                        case 1: bpy.ops.transform.translate(value=(-0.01, 0 , 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, alt_navigation=True)
                        case 2: bpy.context.object.rotation_euler[2] = 0
                else:
                    match Itervalue:
                        case 0: MissingView = "FrontView"
                        case 1: MissingView = "BackView"
                        case 2: MissingView = "SideView"
                    self.report({'ERROR'}, "No inputted Image for" + MissingView)
                Itervalue = Itervalue + 1
            else:
                self.report({'ERROR'}, "No inputted Image.")
                Itervalue = Itervalue + 1

def Feature_detection(self, PlaneDataArray : list[PlaneItem]):
    KeyPoints: list = []
    Descriptors: list = []
    Matches: list = []
    Images: list = []
    Matched_Images: list = []
    ImageNames: list = []
    
    PlaceImage(self, PlaneDataArray) # processes the images

    if(PlaneDataArray.__len__() > 1):
        PlaneIndex = 0
        for PlaneData in PlaneDataArray:
            keypoints1, descriptors1 = detect_and_describe_akaze(PlaneData.PlaneFilepath)
            Images.append(cv2.imread(PlaneData.PlaneFilepath))
            KeyPoints.append(keypoints1)
            Descriptors.append(descriptors1)
            ImageNames.append("MatchedView" + str(PlaneIndex) + PlaneData.PlaneFilepath[PlaneData.PlaneFilepath.rfind("."): ] ) 

        #this should follow this format : #12 #23 #31
        DescriptionIndex = 0
        for descriptors in Descriptors:
            if(DescriptionIndex + 1 != Descriptors.__len__()): NextDesc = Descriptors[DescriptionIndex + 1] #Gets the next descriptor 
            else: NextDesc = Descriptors[0] # if we get to the last index
            Matches.append(match_features(descriptors, NextDesc, method='AKAZE')) 
            DescriptionIndex = DescriptionIndex + 1 

        IndexForKeypoints = 0
        for Keypoint in KeyPoints:
            if(IndexForKeypoints + 1 != KeyPoints.__len__()):
                NextKey = KeyPoints[IndexForKeypoints + 1] #Gets the next descriptor
                NextImage = Images[IndexForKeypoints + 1]
            else:
                NextKey = KeyPoints[0] # if we get to the last index
                NextImage = Images[0]
            Matched_Images.append(draw_matches(Images[IndexForKeypoints], Keypoint, NextImage, NextKey, Matches[IndexForKeypoints]))
            IndexForKeypoints = IndexForKeypoints + 1

        MImageIndex = 0
        for MImages in Matched_Images:
            try:
                os.chdir("Matched_Images_Folder") #changes the directory to the folder where we are going to save the file
                cv2.imwrite(ImageNames[MImageIndex], MImages) #saves the image
                os.chdir("..\\") #goes back one directory   
                print(f"Image prep done.")
                return True
            except Exception as e:
                print(f"Error: {e}")
                return False
# example usage
#image_path = 'C:/Users/RAFAEL MUITO ZIKA/Pictures/emoji disdcord/pekora fate.png'
#prepared_image = prepare_image(image_path)


