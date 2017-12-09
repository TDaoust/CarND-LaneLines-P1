# **Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./test_images_output/innitial.png "Initial"
[image2]: ./test_images_output/canny.png "Canny"
[image3]: ./test_images_output/masked.png "Masked"
[image4]: ./test_images_output/multy_line.png "Canny"
[image5]: ./test_images_output/lines.png "Canny"
[image6]: ./test_images_output/final.png "Canny"


## Reflection:

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

**My pipeline consists of the following steps, with images below for examples:**

1. convert image to gray scale
2. perform canny edge detection (second image)
3. mask the region of interest (third image)
4. use hough line detection to find lines (fourth image)
5. select relevant lines by position and slope, which also separates the left and right lines and discard other lines (fourth image left: white, right: green, discard: red)
6. average lines found for each lane (fifth image)
    1. this also included averaging over multiple frames for the challenge video
    2. using previous frame's lines if there were no acceptable lines found in the current frame (purple line in the video)
7. plotting the averaged line for each lane on the image (sixth image)

**In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:**

1. calculating a slope for each line
2. selecting lines that could belong to the left or right lane marker by slope and region and rejecting other line segments
3. averaging the lines if there were multiple candidates for a lane marker
4. for the challenge video, temporal averaging was implemented across multiple frames to reduce jitter and help reduce the influence of outliers

Images giving an example of how the pipeline works:
![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5] ![alt text][image6]


### 2. Identify potential shortcomings with your current pipeline

1. when the lanes leave the masked region due to lane changes or turning.
2. the rigidly defined slope bounds for the lanes lines, lanes would be missed if you were turn or there was a sharp bend in the road.
3. rigidly define areas where the lanes should be found, same issues as above
4. as seen in the challenge video, when the lines and road are near the same intensity in the gray scale image the lines can be very difficult to extract

### 3. Suggest possible improvements to your pipeline

1. instead of averaging all the line candidates for a lane, use only those that can represent the line, maybe they are grouped of together form a long line
2. using past image(s) and extracted lane(s) as a prior for the current image
3. better smoothing using more advanced techniques than simple averaging

