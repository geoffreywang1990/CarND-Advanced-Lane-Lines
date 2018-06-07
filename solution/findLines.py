import numpy as np
import math
import cv2
import glob
from calibration import calibrate
margin = 100
minpix = 50
invGamma = 1.0/1.5
gammaTable = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

def getBinaryImageframe(img):
	global gammaTable
	assert len(img.shape) == 3
	w = img.shape[1]
	h = img.shape[0]
	#roi= np.zeros_like(img);
	#roi[h//2:h,:,:]= img[h//2:h,:,:]
	cv2.LUT(img,gammaTable)	

	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	gray = hls[:,:,1] 
	s_channel = hls[:,:,2]
	r_channel = img[:,:,0]
	g_channel = img[:,:,1]
	sc_binary = s_select(s_channel,thresh =(170,250))
	r_binary = s_select(r_channel,thresh =(50,255))
	g_binary = s_select(g_channel,thresh =(50,255))
	sx_binary = abs_sobel_thresh(gray,'x',20,100)	
	sy_binary = abs_sobel_thresh(gray,'y',30,100)	
	mag_binary = mag_thresh(gray,3,(50,110))	
	dir_binary = dir_threshold(gray,15,(0.7,1.3))
	combined_binary = np.zeros_like(sx_binary)
	#combined_binary[(sc_binary == 1) | (sx_binary == 1) | (mag_binary == 1) ]= 1 #| ((mag_binary == 1) )] = 1
	#combined_binary[(sc_binary == 1) | (mag_binary ==1) ]= 1
	combined = sc_binary * + sx_binary *10 + sy_binary * 5 + mag_binary * 10 + dir_binary*10
	#combined_binary[combined > 20]= 1 #| ((mag_binary == 1) )] = 1
	combined_binary[ ((sc_binary == 1) & (r_binary ==1) & (g_binary ==1) )| (mag_binary == 1) ]= 1 #] = 1
	return combined_binary 
	#return contours_select(combined_binary)

def contours_select(binary):
	w = binary.shape[1]
	h = binary.shape[0]
	ret = np.zeros_like(binary)
	longLine = 0.4 * h
	minRegionSize = 0.00001 * w * h
	smallLaneArea = 5 * minRegionSize
	ratio =3 
	leftLinePos = 2 * w //5
	rightLinePos = 3 * w //5
	_, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		if cv2.contourArea(contour) < minRegionSize:
			continue
		rotrect = cv2.minAreaRect(contour)
		#print(rotrect)
		ct_w = rotrect[1][0]
		ct_h = rotrect[1][1]
		if ct_w/ct_h<3  and ct_h/ct_w <3:
			continue
		ct_angle = abs(rotrect[2])
		if ((ct_w > longLine) or (ct_h > longLine))  :
			cv2.fillPoly(ret,[contour],1,0)
		elif ((ct_angle > 20) and ( ct_angle < 70 )) and ((rotrect[0][0] > leftLinePos) or (rotrect[0][0] <rightLinePos)):
			cv2.fillPoly(ret,[contour],1,0)
	return ret

def s_select(s_channel, thresh=(0, 255)):
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
	
	# Apply the following steps to img
	# 1) Convert to grayscale
	# 2) Take the derivative in x or y given orient = 'x' or 'y'
	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	else:
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	# 3) Take the absolute value of the derivative or gradient
	abs_sobel = np.absolute(sobel)
	
	# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# 5) Create a mask of 1's where the scaled gradient magnitude 
	        # is > thresh_min and < thresh_max
	sxbinary = np.zeros_like(scaled_sobel)

	sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1 
	
	# 6) Return this mask as your binary_output image
	
	return sxbinary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
	
	# Apply the following steps to img
	# 1) Convert to grayscale
	
	
	# 2) Take the gradient in x and y separately\
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	# 3) Calculate the magnitude 
	abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
	# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# 5) Create a binary mask where mag thresholds are met
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	
	return sxbinary


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):

	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	# 3) Take the absolute value of the x and y gradients
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	
	# 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
	gradient = np.arctan2(abs_sobely, abs_sobelx)
	# 5) Create a binary mask where direction thresholds are met
	sxbinary = np.zeros_like(gradient)
	sxbinary[(gradient >= thresh[0]) & (gradient <= thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	return sxbinary



def compute_perspective_transform(w,h):
	transform_src = np.float32([ [615,450], [340,645], [1000,645], [700,450]])
	transform_dst = np.float32([ [300,100], [300,700], [980,700], [980,100]])
	M = cv2.getPerspectiveTransform(transform_src, transform_dst)
	invM = cv2.getPerspectiveTransform(transform_dst, transform_src)
	return [M,invM]


def apply_perspective_transform(binary_image, M):
	warped_image = cv2.warpPerspective(binary_image, M, (binary_image.shape[1], binary_image.shape[0]), flags=cv2.INTER_NEAREST)  
	#warped_image[:,:300]= 0
	#warped_image[:,1080:]= 0
	#warped_image[:,350:930]= 0
	

	#return contours_select(warped_image)
	
	#return contours_select(warped_image,1,(0,30),(60,90))
	return warped_image


def identifyLines(binary_warped,previous_l=[],previous_r=[]):
	global margin,minpix
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	if previous_r == [] and previous_l == []: 
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]//2)
		leftx_base = np.argmax(histogram[:midpoint-binary_warped.shape[1]//7])
		rightx_base = np.argmax(histogram[midpoint + binary_warped.shape[1]//7:]) + midpoint + binary_warped.shape[1]//7
		
		# Choose the number of sliding windows
		nwindows = 9
		# Set height of windows
		window_height = np.int(binary_warped.shape[0]//nwindows)
		# Identify the x and y positions of all nonzero pixels in the image

		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		# Set minimum number of pixels found to recenter window
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []
		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			#if previous_l != []:
			#	leftx_current = int(sum(previous_l[win_y_low:win_y_high])/float(win_y_high-win_y_low))
			#if previous_r !=[]:
			#	rightx_current = int(sum(previous_r[win_y_low:win_y_high])/float(win_y_high-win_y_low))
			#print(leftx_current,rightx_current)
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
			(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
			(0,255,0), 2) 
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
	else:
		left_lane_inds = ((nonzerox > (previous_l[0]*(nonzeroy**2) + previous_l[1]*nonzeroy + 
						previous_l[2] - margin)) & (nonzerox < (previous_l[0]*(nonzeroy**2) + 
							previous_l[1]*nonzeroy + previous_l[2] + margin))) 

		right_lane_inds = ((nonzerox > (previous_r[0]*(nonzeroy**2) + previous_r[1]*nonzeroy + 
						previous_r[2] - margin)) & (nonzerox < (previous_r[0]*(nonzeroy**2) + 
							previous_r[1]*nonzeroy + previous_r[2] + margin)))  

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	return[ left_fit, right_fit, ploty, left_fitx, right_fitx,  leftx, lefty, rightx, righty]




def draw_lane( undist, warped, invM, left_fitx, right_fitx, ploty):

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (invM)
    newwarp = cv2.warpPerspective(color_warp, invM, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def compute_curvature( left_fit, right_fit, ploty, left_fitx, right_fitx, leftx, lefty, rightx, righty):
        
    ym_per_pix = 20/600 # meters per pixel in y dimension
    xm_per_pix = 3.7/780 # meters per pixel in x dimension
    
    y_eval = np.max(ploty)

    fit_cr_left = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    curve_l= ((1 + (2 * left_fit[0] * y_eval / 2. + fit_cr_left[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_left[0])
    fit_cr_right = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    curve_r= ((1 + (2 * right_fit[0] * y_eval / 2. + fit_cr_right[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_right[0])

    return (curve_l+ curve_r) / 2

def compute_center_offset(left_fitx,right_fitx, undist_image):
	    
	xm_per_pix = 3.7/780
	lane_center_x = int(np.average((left_fitx[180:]+right_fitx[180:])/2))
	image_center_x = int(undist_image.shape[1] / 2)
	offset_from_center = (image_center_x - lane_center_x) * xm_per_pix 
	
	return offset_from_center


def render_curvature_and_offset(undist_image, curve, offset):

   offset_text = 'offset is: {:.2f}m'.format(offset)
   font = cv2.FONT_HERSHEY_SIMPLEX
   cv2.putText(undist_image, offset_text, (30, 50), font, 1, (255, 255, 255), 2)

   curve_text = 'curverature is: {:.2f}m'.format(curve)
   cv2.putText(undist_image, curve_text, (19, 90), font, 1, (255, 255, 255), 2)

   return undist_image

if __name__=="__main__":
	mtx,dist,mapx,mapy = calibrate(True)
	img = cv2.imread("../test_images/test1.jpg")
	color = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
	
	cv2.imwrite('../writeup_img/test1.jpg',color)
	binary = getBinaryImageframe(color)
	cv2.imwrite('../writeup_img/binary_combo.jpg',binary*255)

	w = img.shape[1]
	h = img.shape[0]
	M,invM = compute_perspective_transform(color.shape[1],color.shape[0])
	perspective_binary =apply_perspective_transform(binary,M)
	cv2.imwrite('../writeup_img/warped_straight_lines.jpg',perspective_binary*255)
	left_fit, right_fit, ploty, left_fitx, right_fitx,  leftx, lefty, rightx, righty = identifyLines(perspective_binary)
	
	out_img = np.dstack((perspective_binary, perspective_binary, perspective_binary))*255
	window_img = np.zeros_like(out_img)
	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
	                              ploty])))])

	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
	                              ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))
	
	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	cv2.imwrite('../writeup_img/color_fit_lines.jpg',result)
	ret2 = draw_lane(color,perspective_binary,invM,left_fitx,right_fitx,ploty)
	cv2.imwrite('../writeup_img/example_output.jpg',ret2)
	cv2.imshow("binary",perspective_binary*255)
	cv2.imshow("final",ret2)
	cv2.waitKey()
	cv2.destroyAllWindows()
