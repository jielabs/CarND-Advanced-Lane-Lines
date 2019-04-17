## Writeup 

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

[image1]: ./examples/1.png "chessboard points"
[image2]: ./examples/2.png "undistoted image"
[image3]: ./examples/3.png "distortion correction"
[image4]: ./examples/4rgb.png "RGB bands"
[image5]: ./examples/5hls.png "HLS bands"
[image6]: ./examples/6.png "Sobel on X"
[image7]: ./examples/7.png "Sobel on Y"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Writeup / README

### Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All the codes are in the Jupyter notebook "Project_Pipeline.ipynb". I wil use cell number as the reference of code locatoin. There are 26 cells in total.

If you cannot open Project_Pipeline.ipynb, please use "Project_Pipeline.pdf" or "Project_Pipeline.html" instead.

---

## Camera Calibration

### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell 3 and 4.

To show the points:
  1. Convert RGB image to gray image which is easier to process.
  1. Find the chessboard corners using `cv2.findChessboardCorners`
  1. Draw the chessboard corners on the image.
  1. Save all the corners in `img_points`, and all the coordinates in `obj_points`.

![chessboard points][image1]

To generate undistorted images:
  1. Call `cv2.calibrateCamera(obj_points, img_points, ...)` to get `mtx` and `dist`.
  1. Call `cv2.undistort(image, mtx, dist, None, mtx)` to get the undistorted image.
  1. Show both orginal and undistored image side by side.

![undistoted image][image2]  

---

## Pipeline (test images)

### 1. Provide an example of a distortion-corrected image.

Cell 5 has the code to apply a distortion correction on a RGB image. Basically it calls `cv2.undistort(test_image, mtx, dist, None, mtx)` to convert the image. The `mtx` and `dist` are obtained from the previous step.

![undistoted rgb image][image3]

I applied the distortion correction on all the images and save them in a list. In the following tasks, I always use undistorted image as the input RGB image.


### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

**Color space**
In Cell 6 I converted a RGB image to HLS space, and then checked the all the bands individually.

RGB bands
![rgb bands][image4]

HLS bands
![hls bands][image5]

It seems the S bands can highlight the line much more clearly than other bands. R band is the second best one. Therefore I will only use S band for the thresholding later.

**Sobel operation**

I used Sobel on S band on X direction and Y direction in cell 7, using `cv2.Sobel(s_band, cv2.CV_64F, xorder, yorder, ksize=sobel_kernel)`, followed by getting the absolute value. I tuned the kernal size a bit and decide to keep it 3. You can find the results below:

![sobel-6][image6]

**Gradient on Sobel**

In cell 12, I computed the scale of sobel gradient using euclidean distance `np.sqrt(sobelx**2 + sobely**2)`. The result is below:

![sobel-7][image7]

In cell 13, I computed the angle of sobel gradient, and tried to tune the threshold. The result is below:

[image8]: ./examples/8.png "Sobel angle"
![sobel-8][image8]

**Combined mask**

Finally, in cell 14 I combined all kinds of masks with the tuned thresholds using `combined[((sobel_x == 1) & (sobel_y == 1)) | ((grad == 1) & (grad_direction == 1))] = 1`.

[image9]: ./examples/9.png "combined mask"
![combined mask][image9]


### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I applied the perspective transform to rectify binary image in cell 15 - 18.

First, I found a polygon which covers the two lines in front of the car in cell 15.

[image10]: ./examples/10.png
![][image10]

In cell 16, I set the target shape as a rectangle, and the polygon I got before as the source, so I can call cv2.getPerspectiveTransform to compute the transformation. Then I called `cv2.warpPerspective(undist_img, M, img_size)` to compute the warped image.

[image11]: ./examples/11.png
![][image11]

In cell 17-18, I applied the warpPerspective to the combined mask as well. Here is what I got:

[image12]: ./examples/12.png
![][image12]

[image13]: ./examples/13.png
![][image13]


### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In cell 19, I wrote a function to get the histogram from the bottom half of the mask. It demonstrated the idea of how to get the position of line on x dimention, because there are two clear peaks in the histogram.

[image14]: ./examples/14.png
![][image14]

[image15]: ./examples/15.png
![][image15]

Cell 20 has the most complex codes. Basically I followed the suggestion from the course, trace the x poistion on moving windows vertically, and fit a line using polyfit. All those operations are done on warped mask, and I did it on left and right side independently.

I set the minimum number of points inside a moving window to be qualified as good points, save the coordinates of all the good pixels, and fit the curve on them.

When there are not many points, the fitting may not work very well.

The results are shown in cell 21.

[image16]: ./examples/16.png
![][image16]

[image17]: ./examples/17.png
![][image17]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In cell 22, I computed the curvature based on the fitted curves. I also got the fitted curve on meters (assuming the meters per pixel is fixed. In y dimension it is ym_per_pix = 30/720, and xm_per_pix = 3.7/700 for x dimension).

The formula to compute the curvature is (in km)
  * `left_curverad = ((1 + (2*left_fit_m[0]*y_eval*ym_per_pix + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0]) / 1000`
  * `right_curverad = ((1 + (2*right_fit_m[0]*y_eval*ym_per_pix + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0]) / 1000`

Results on 2 test images:

Curvature of image 6 : **left 0.76 km, right 0.60 km**

[image18]: ./examples/18.png
![][image18]

Curvature of image 7 : **left 0.83 km, right 0.43 km**

[image19]: ./examples/19.png
![][image19]


### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In cell 24, I created a function `draw_lane()` to annotated the lane using a green mask. 

To get this annotaton (the green polygon), I first need to fill the polygon between the two fitted curves on the warp mask, then apply `cv2.warpPerspective` using `Minv`, the reverted perspective transformation on the polygon, and add it back to the original image.

[image20]: ./examples/20.png
![][image20]

[image21]: ./examples/21.png
![][image21]

I also displayed the main metrics (such as curvature and location to the center) on the image, with an optional flag `show=True` to function `draw_lane()`.

[image23]: ./examples/23.png
![][image23]

---

## Pipeline (video)

### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Finally, I applied the lane mask on all the images of the video, and generate a video with the annotated lane.

Here's a [link to the annotated video result](./annotated_video.mp4) (annotated_video.mp4)

I also uploaded it to **Youtube**:
[![Annotated Lane](https://i.ytimg.com/vi/w2Ws687C2AI/hqdefault.jpg?sqp=-oaymwEjCNACELwBSFryq4qpAxUIARUAAAAAGAElAADIQj0AgKJDeAE=&rs=AOn4CLBl1Nt5kCiPz92tGsIAimysD5bVzQ)](https://youtu.be/w2Ws687C2AI "Annotated Lane")



---

## Discussion

### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Tuning parameters is really hard and time consuming. One setting may work well for one set of images, but may not work for others. 

CV2 libraries are powerful, but learning them are not easy. I spent lots of time to fix a bug, which turned out I used a wrong order to get the height and width from image.shape. 

I hope to use ML based approach to do computer vision tasks, and use CV based approach for simple tasks.

