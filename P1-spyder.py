#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 21:29:30 2017

@author: rtuser
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import os
#image_name_list = os.listdir("test_images/")
image_name_list = os.listdir("testing/")
for image_name in image_name_list:
    print(image_name)
    
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def extrap_x(y,b,m):
    """
    `y` is the y pixel of the line.
    
    `b` is the y intercept of the line.
    
    `m` is the slope of the line.
        
    Returns the corresponding point x for the given line parameters.
    """
    x = ((y-b)/m)
    
    return int(x)

def mean_line(l_points,img_shape):
    
    x1,y1,x2,y2 = np.mean(l_points,axis=0)
    
    m = ((y2-y1)/(x2-x1))
    b = (y1 - (m * x1))
    
    y_bot = img_shape[0]
    y_top = int(img_shape[0]*.65)
    
    p_bot = (extrap_x(y_bot,b,m),y_bot)
    p_top = (extrap_x(y_top,b,m),y_top)
    
    return p_bot,p_top

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    im_x = img.shape[1]
    left_line = []
    right_line = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = ((y2-y1)/(x2-x1))
            if m > -0.9 and m < -0.4 and x1 < im_x * 0.6 and x2 < im_x * 0.6:
                left_line.append([m,x1,y1,x2,y2])
                #cv2.line(img, (x1, y1), (x2, y2), [255,255,255], 2)
                #print('Left: m = '+ str(m) + ' x1 = '+ str(x1) + ' x2 = '+ str(x2))
            elif m > 0.4 and m < 0.9 and x1 > im_x * 0.4 and x2 > im_x * 0.4:
                right_line.append([m,x1,y1,x2,y2])
                #cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
                #print('Right: m = '+ str(m) + ' x1 = '+ str(x1) + ' x2 = '+ str(x2))
            #else:
                #print('Fail: m = '+ str(m) + ' x1 = '+ str(x1) + ' x2 = '+ str(x2))
                #cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
                
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    left_line = np.array(left_line)
    right_line = np.array(right_line)
    
    #print('left:',left_line.shape ,' right: ', right_line.shape) 
    
    both_lines = True
    
    if left_line.shape[0] !=0:
        p_bot,p_top = mean_line(left_line[:,1:],img.shape)
        cv2.line(img, p_bot, p_top, [255, 0, 0], thickness)
    else:
        print('No left lines')
        both_lines = False
    
    if right_line.shape[0] !=0:
        p_bot,p_top = mean_line(right_line[:,1:],img.shape)
        cv2.line(img, p_bot, p_top, [255, 0, 0], thickness)
    else:
        print('No right lines')
        both_lines = False
    return both_lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    both_lines = draw_lines(line_img, lines)
    return line_img, both_lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=0.8, l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, l)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
#for image_name in image_name_list:

gauss_kernel_size = 7 #odd numbers only
canny_low_threshold = 80
canny_high_threshold = 120

#if 1:
for image_name in image_name_list:
#    image_name = image_name_list[5]
    #image = mpimg.imread('test_images/'+image_name)
    image = mpimg.imread('testing/'+image_name)
    print('This image is:', type(image), 'with dimensions:', image.shape)
    #print(image_name)
    plt.figure()
    plt.title(image_name)
    plt.imshow(image)
    
    
    
    #plt.figure()
    #plt.title(image_name+' grey_blur')
    #plt.imshow(blur_gray, cmap='Greys_r')
    
    edges = cv2.Canny(gaussian_blur(grayscale(image),gauss_kernel_size), canny_low_threshold, canny_high_threshold)
    
    plt.figure()
    plt.title(image_name+' edges')
    plt.imshow(edges, cmap='Greys_r')
    
    #bot_left = (50,image.shape[0])
    #top_left = (410, 340)
    #top_right = (550, 340)
    #bot_right = (920,image.shape[0])
    #vertices = np.array([[bot_left,top_left, top_right, bot_right]], dtype=np.int32)
    
    im_x = image.shape[1]
    im_y = image.shape[0]
    
    bot_left = (im_x*0.04,im_y)
    top_left = (im_x*0.40, im_y*0.61)
    top_right = (im_x*0.58, im_y*0.61)
    bot_right = (im_x*0.95,im_y)
    vertices = np.array([[bot_left,top_left, top_right, bot_right]], dtype=np.int32)
    
    masked_image = region_of_interest(edges, vertices)
    
    plt.figure()
    plt.title(image_name+' mask')
    plt.imshow(masked_image, cmap='Greys_r')
    
    hough_rho = 1 # distance resolution in pixels of the Hough grid
    hough_theta = np.pi/180 # angular resolution in radians of the Hough grid
    hough_threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    hough_min_line_length = 10 #minimum number of pixels making up a line
    hough_max_line_gap = 150    # maximum gap in pixels between connectable line segments
    
    line_img,both_lines = hough_lines(masked_image, hough_rho, hough_theta, hough_threshold, hough_min_line_length, hough_max_line_gap)
    
    print(image[:,:,1:].shape,line_img.shape)
    
    print('This line is:', type(line_img), 'with dimensions:', line_img.shape)
    
    plt.figure()
    plt.title(image_name+' lines')
    plt.imshow(line_img)
    
    plt.figure()
    plt.title(image_name+' final')
    plt.imshow(weighted_img(line_img, image[:,:,:3]))
    
    #mpimg.imsave('test_images_output/'+image_name,weighted_img(line_img, image))
    
    
    