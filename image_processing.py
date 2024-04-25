import cv2
from .image_tracker import CustomMultiTracker
import os
import numpy as np
import bpy
import random
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
 
# def angle_between(p1, p2, p3):
#     
#     a = np.array(p1)
#     b = np.array(p2)
#     c = np.array(p3)

#     # did i really just use math here
#     ba = a - b
#     bc = c - b

#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.arccos(cosine_angle)

#     return np.degrees(angle)    
    
# TODO: refactor image prep function to utilize harris corner detection
# a bunch of repeated code. 
def find_and_color_vertices(image_path):
    
    # load image
    image = cv2.imread(image_path)

    # repeated code tbh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # helps vizualizing corners
    dst = cv2.dilate(dst, None)

    # threshold for a corner. using default values
    corners_threshold = dst > 0.01 * dst.max()

    # getting the indices of corner points
    corners = np.argwhere(corners_threshold)

    corners = np.flip(corners, axis=1) # inverting y and x coordinates because np.argwhere returns in (row, column) format

    num_corners = len(corners)

    # drawing corners on the image
    for x, y in corners:
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)


    print(f'Number of corners found: {num_corners}')

    cv2.imshow('Detected Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return corners
    
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
                    FileDirectory = plane_data.ImagePlaneFilePath[: plane_data.ImagePlaneFilePath.rfind("\\")] + "\\"
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

def camera_corner(): 
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # lower and upper definition of red
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # mask for range of  colors
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2  # Combine masks for red color

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filter out small contours
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if 6 <= len(approx) <= 10:
                    draw_cube(approx, frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def camera_test():
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # defining colors/range for mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # remove noise with morph
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # consider only large enough areas

                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # draw the corners and contour
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)  # green contour
                for point in approx:
                    x, y = point.ravel()
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # ded corners

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)  

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return 0

# function to handle drawing the cube
def draw_cube(approx, frame):
    if len(approx) == 8:  # Expecting 8 corners for a typical cube projection
        approx = sorted(approx, key=lambda x: (x[0][0], x[0][1]))
        # connect corners
        for i in range(4):
            x, y = approx[i][0] # need to revisit this.
            x_next, y_next = approx[(i + 1) % 4][0]
            cv2.line(frame, (x, y), (x_next, y_next), (255, 0, 0), 2)
            # connect vertical edges of the cube
            cv2.line(frame, (x, y), approx[i + 4][0], (255, 0, 0), 2)
        # connect top edges
        for i in range(4, 8):
            x, y = approx[i][0]
            x_next, y_next = approx[i + 1 if i < 7 else 4][0]
            cv2.line(frame, (x, y), (x_next, y_next), (255, 0, 0), 2)

# function to check if the detected contour could be a face of the cube
def is_potential_cube_face(contour, min_area=500, aspect_ratio_range=(0.8, 1.2)):
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    return area > min_area and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]

def initialize_tracker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # two ranges of hsv red colors
    lower_red1 = np.array([0, 70, 50])   # Lower end of the hue for red (0-10)
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]) # Higher end of the hue for red (170-180)
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # Combine masks for the full range of red

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour, mask
    return None, mask

def update_tracker(frame, tracked_contour):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour, mask
    return None, mask

def find_cube_corners(frame):
    
    # image prep
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  

    # edge and countour detecction
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    for cnt in contours:
        # approx contour to reduce the number of points. unsure if needed
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # We expect a cube to have many corners depending on the perspective
        if len(approx) >= 4:  # we only want 4 bc we are working with a cube specifically.
            for p in approx:
                corners.append(tuple(p[0]))  # p[0] contains x, y cords

    return corners

# testing using trackers instead of feature detectors
def testing_tracker():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        return
    
    tracked_contour, mask = initialize_tracker(frame)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if tracked_contour is not None:
            tracked_contour, mask = update_tracker(frame, tracked_contour)
            if tracked_contour is not None:
                cv2.drawContours(frame, [tracked_contour], -1, (0, 255, 0), 3)
        
        cv2.imshow('Tracker', frame)
        cv2.imshow('Mask', mask)  # Display the mask to see what's being detected
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

#
def test_shit():
    
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # red color
        low_red = np.array([161, 155, 84])
        high_red = np.array([179, 255, 255])
        red_mask = cv2.inRange(hsv_frame, low_red, high_red)
        red = cv2.bitwise_and(frame, frame, mask=red_mask)

        # countours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)
                # Use the vertices 'approx' as needed

        cv2.imshow("Frame", frame)
        cv2.imshow("Red", red)


        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import time
    
def cringe_ai_model():
    # testing this cringe code that i barely understand
    cap = cv2.VideoCapture(0)
    model = load_model()
    transform = get_transform()
    time_threshold = 50  # Process 1 frame every 5 seconds
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - last_time

        # i am limiting frames per second so my pc doesnt explode. not even sure if needed
        if elapsed_time > time_threshold:
            vertices = find_vertices(frame)
            depth_map = estimate_depth(model, transform, frame)

            for vertex in vertices:
                for point in vertex:
                    x, y = point.ravel()
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    depth_value = depth_map[y, x].item()
                    cv2.putText(frame, f"{depth_value:.2f}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow('Processed Frame', frame)
            last_time = current_time

        cv2.imshow('Live Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

# load the MiDaS model for depth estimation
def load_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model

# Transform for depth estimation model input
def get_transform():
    return Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def estimate_depth(model, transform, frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        depth = model(input_tensor)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(frame.shape[0], frame.shape[1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return depth

def find_vertices(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    red_mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)
    red_mask = cv2.erode(red_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vertices_list = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices_list.append(approx)
    return vertices_list
