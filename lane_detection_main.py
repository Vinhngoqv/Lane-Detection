import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def make_coordinates(image, line_parameters):                                   # Function create points of the lane for visulization
    slope, intercept = line_parameters
    width = image.shape[1]
    y1 = image.shape[0]
    y2 = int(y1*0.75)

    result = np.empty(shape=[0,2]).astype(int)
    for y_temp in range(y2,y1,8):                                               # Adjust the rate of points in the lane line
        x = int((y_temp - intercept)/slope)
        if abs(x) > width:
            x = width
        result = np.append(result, [[x, y_temp]], axis=0)
    return result

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < -0.5:                                                       # Adjust the slope rate of the left line
            left_fit.append((slope, intercept))
        elif slope > 0.5:                                                       # Adjust the slope rate of the right line
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0) 

    if right_fit != [] and left_fit != []:
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.append(left_line, right_line, axis=0)
    elif right_fit != [] and left_fit == []:
        right_line = make_coordinates(image, right_fit_average)
        return right_line
    elif right_fit == [] and left_fit != []:
        left_line = make_coordinates(image, left_fit_average)
        return left_line


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 150,160)                                                    # Adjust the threshold for Canny convertion
    return canny 

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    width = image.shape[1]
    if lines is not None:
        for line in lines:
            x, y = line.reshape(2)
            
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
            # cv2.line(line_image, (x1, y1), (x2, y2), (0,0,255), 10)
    return line_image

def region_of_interest(image):
    height, width = image.shape[:2]

    ### Create a big mask image for RoI processing
    point_A = (int(width*0.15),height)                                                    # Left bottom of the big polygon
    point_B = (int(width*0.7),height)                                                    # Right bottom of the big polygon
    point_C = (int(width*0.6),int(height*0.73))                                            # Right top of the big polygon
    point_D = (int(width*0.45),int(height*0.73))                                            # Left top of the big polygon

    polygons = np.array([[point_A,point_B,point_C,point_D]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    ### Create a small mask image for RoI processing
    gap_offset = 150                                                                         # The distance of RoI in pixel
    point_E = (point_A[0]+gap_offset,point_A[1])                                            # Left bottom of the small polygon
    point_F = (point_B[0]-gap_offset+30,point_B[1])                                            # Right bottom of the small polygon
    point_G = (point_C[0]-gap_offset+40,point_C[1]+100)                                            # Right top of the small polygon
    point_H = (point_D[0]+gap_offset-60,point_D[1]+100)                                            # Left top of the small polygon

    polygons_sm = np.array([[point_E,point_F,point_G,point_H]])
    mask_sm = np.zeros_like(image)
    cv2.fillPoly(mask_sm, polygons_sm, 255)
    masked_image = cv2.bitwise_xor(mask,mask_sm)
    RoI_image = cv2.bitwise_and(image, masked_image)
    return RoI_image

def save_video_result(image):
    height,width,_ = image.shape
    save_path = os.path.join('Result',video_path[0:-4]+'_result.mp4')
    print(save_path)
    result = cv2.VideoWriter(save_path,-1,20,(width,height))
    result.write(image)

video_path = r'Lane Detection Test Video 01.mp4'
cap = cv2.VideoCapture(video_path)
while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    lane_image = np.copy(frame)
    img_canny = canny(frame) 
    cropped_image = region_of_interest(img_canny)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 40, np.array([]), minLineLength=50, maxLineGap=100) 

    # cv2.imshow('Lane Detection',cropped_image)
    # cv2.imshow('Original lane', frame)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

    ## Visualize the Hough Line function 
    if lines is not None:
        # for lline in lines:
        #     x1, y1, x2, y2 = lline.reshape(4)
        #     cv2.line(lane_image,(x1,y1),(x2,y2),(0,255,0),2)

        averaged_lines = average_slope_intercept(lane_image, lines)
        line_image = display_lines(lane_image,averaged_lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

        cv2.imshow('Lane Detection',lane_image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()