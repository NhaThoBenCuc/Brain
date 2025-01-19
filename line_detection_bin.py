import numpy as np

def non_max_suppression(lines, threshold):
    """
    Perform non maximum suppression on a list of lines.
    """
    lines = np.array(lines)
    if len(lines) == 0:
        return lines

    # Compute the angle of each line
    angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])

    # Sort lines by angle
    sorted_lines = lines[np.argsort(angles)]

    # Initialize the list of picked lines
    picked_lines = []

    # Perform non maximum suppression
    for i in range(len(sorted_lines)):
        if i == 0:
            picked_lines.append(sorted_lines[i])
            continue

        # Compute the angle difference between the current line and the last picked line
        angle_diff = np.abs(np.arctan2(sorted_lines[i][3] - picked_lines[-1][1], sorted_lines[i][2] - picked_lines[-1][0]))

        # If the angle difference is greater than the threshold, add the line to the list of picked lines
        if angle_diff > threshold:
            picked_lines.append(sorted_lines[i])

    return np.array(picked_lines)

import cv2
from tqdm import tqdm
def bird_eye_view(image):
    IMAGE_H = 1232
    IMAGE_W = 1640

    src = np.float32([[150, IMAGE_H], [1500, IMAGE_H], [600, 650], [1150, 650]])
    dst = np.float32([[600, IMAGE_H], [1150, IMAGE_H], [600, 650], [1150, 650]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    # img = cv2.imread('./test_img.jpg') # Read the test img
    # img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop

    warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H)) # Image warping

    return warped_img
def get_angle(lines):
    """
    Compute angle of vertical lines.
    """
    # horizontal_threshold = 50
    # horizontal_lines = [line for line in lines if abs(line[0][0] - line[0][2])>horizontal_threshold]
    vertical_threshold = 50
    if lines is None:
        return 0
    vertical_lines = [line for line in lines if abs(line[1] - line[3])>vertical_threshold]
    if len(vertical_lines) == 0:
        return 0
    res = 0
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        #get alpha angle
        alpha=abs(np.arctan2(y2-y1, x2-x1))
        res+=alpha/np.pi*180

    res = res/len(vertical_lines)
    return res
def detect_lanes(image):
    global total_lines
    # Convert to grayscale
    image = bird_eye_view(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = bird_eye_view(gray)
    median = gray.copy()
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(median, (5, 5), 0)

    # min max normalization
    norm_img = (blurred - blurred.min()) / (blurred.max() - blurred.min()) * 255
    
    # Put all intensity into 10 bins
    num_bin = 10
    bins = np.linspace(0, 255, num_bin)
    digitized = np.digitize(norm_img, bins) - 1

    # Make pixels in all bins except the last one black
    gray[digitized < num_bin-2] = 0
    binned = gray.copy()
    binned = binned.astype(np.uint8)
    # Perform Canny edge detection
    edges = cv2.Canny(binned, 50, 150)

    # Define region of interest
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width, height // 2),(width // 2,0),(0,height//2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    # Apply Hough transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=10)

    # non maximum suppression
    if lines is not None:
        lines = np.array(lines).reshape(-1, 4)
        lines = non_max_suppression(lines, 0.1)

    # Classify lines as dashed or continuous
    if lines is not None:
        total_lines += len(lines)
        for line in lines:
            x1, y1, x2, y2 = line
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if line_length < 100:  
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
            else:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

    return image, get_angle(lines)

total_lines = 0
# Open the video file
vid_name = 'test'
video_path = f"./Records/{vid_name}.avi"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
print(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'./output/{vid_name}.mp4', fourcc, fps, (width, height))

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Process each frame with a progress bar
result = []
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)
    if not ret:
        break

    # Process the frame to detect lanes
    processed_frame,angle = detect_lanes(frame)
    # result.append(processed_frame)
    # cv2.imshow('frame', processed_frame)
    # cv2.waitKey(1)
    # Write the processed frame to the output video
    out.write(processed_frame)
# Release `the video capture and writer objects
print("Average lines detected:", total_lines//total_frames)
print()
cap.release()
out.release()