## Writeup 

### Yujun Wang

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_img/undistort_output.jpg "Undistorted"
[image2]: ./writeup_img/test1.jpg "Road Transformed"
[image3]: ./writeup_img/binary_combo_example.jpg "Binary Example"
[image4]: ./writeup_img/warped_straight_lines.jpg "Warp Example"
[image5]: ./writeup_img/color_fit_lines.jpg "Fit Visual"
[image6]: ./writeup_img/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the python script file located in "./solution/calibration.py" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.initUndistortRectifyMap()` and ç to obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
to get such an undistortion image, I will apply the `mapx` and `mapy` calculated from the last step to the test images with `cv2.imread()` function.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of S channel selection and soble x direction to generate a binary image (thresholding steps at lines #17 through #18 in `findLines.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes two functions called `compute_perspective_transform()` and `apply_perspective_transform()`, which appears in lines 94 through 104 in the file `findLines.py`. The `compute_perspective_transform()` function takes as width(`w`) and height(`h`) of input image and output the transform matrix(`M`). The `apply_perspective_transform()` function takes a binary image(`binary_image`) and transform Matrix(`M`) as input, and output the projected image. I chose the hardcode the source and destination points by selecting points from one of the undistorted test image, and projected them to a rectangle.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 615, 450      | 340, 400        | 
| 340, 645      | 340, 670      |
| 1000, 645     | 1000, 670      |
| 700, 450      | 1000, 400        |

I verified that my perspective transform was working as expected by projecting a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this: go over the image row by row, find the peak of the 1D signal as left and right line marks on the left and right side of the image accordingly.

After got the line marks, I fitted the lane with `numpy.polyfit()` function.
The code is located at `findlines.py` from line 106 to line 189.


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #208 through #228 in my code in `findLines.py`. I assume in y direction, the FOV is about 30 meters in the projected image, so each pixel is 30/720 meters. in x direction, assuming the FOV is 4 meters, so each pixel is 4/1280 meters.
Calucate the radius of the curvature based on the math from [wikipedia](https://en.wikipedia.org/wiki/Radius_of_curvature).
Calucate the offset of the vehicle by assuming the image center is the center of the ego vehicle. Calculate the difference between image center and lane center, multiply by meter per pixel in x direction. 



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #189 through #206 in my code in `findLines.py` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issue I am facing in this project is the shading in the image. Shadings in gradient somehow is simlar with line marks. Similarly, bad road conditions also looks like a road marks to the algorithm.
One potential solution for this problem is using info from last frame. Lanes between two frames shouldn't differ too much.

