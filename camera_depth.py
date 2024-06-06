import cv2
import numpy as np


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