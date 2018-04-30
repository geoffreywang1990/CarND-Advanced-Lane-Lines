import numpy as np
import cv2
import glob
from calibration import calibrate
def getBinaryImageframe(img):
	assert len(img.shape) == 3
	w = img.shape[1]
	h = img.shape[0]
	roi= np.zeros_like(img);
	roi[h//2:h,:,:]= img[h//2:h,:,:]

	hls = cv2.cvtColor(roi, cv2.COLOR_RGB2HLS)
	gray = hls[:,:,1] 
	s_channel = hls[:,:,2]
	sc_binary = s_select(s_channel,thresh =(170,255))
	sx_binary = abs_sobel_thresh(gray,'x',20,100)	
	combined_binary = np.zeros_like(sx_binary)
	combined_binary[(sc_binary == 1) | (sx_binary == 1)] = 1
	return combined_binary


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


def deNoise(binary_img)
	w = binary_img.shape[1]
	h = binary_img.shape[0]
	retImg  = np.zeros_like(binary_img)
	longLine = 0.2*h
	minRegionSize = 0.002 * w * h
	smallLaneArea = 5 * minRegionSize
	ratio = 4

	leftLine = 2 * frame.cols // 5
	rightLine = 3 * frame.cols // 5
	# find countours in binary image	
	#cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) ¡ú contours, hierarchy
	contours, hierarchy = cv2.findContours(binary_img, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE)
	for ct in countours:
		ct_area = cv2.contourArea(ct)
		if ct_area > minRegionSize:
			rotated_rect = cv2.minAreaRect(ct)
			w = rotated_rect.size[1]
			h = rotated_rect.size[2]
			theta = rotated_rect.angle
		
			if(w > h):
				theta  = 90 + theta
			if h >longLine or w > longLine:
				cv2.fi	
				
'''
                    drawContours(frame, contours,i, cvScalar(0), CV_FILLED, 8);
                    cv::Vec4f line;
                    cv::fitLine(contours[i], line, CV_DIST_L2, 0, 0.01, 0.01);
                    
                    float vx,vy,x,y;
                    vx = line[0];
                    vy = line[1];
                    x =line[2];
                    y=line[3];
                    int lefty,righty;
                    
                    lefty = int((-x*vy/vx) + y);
                    righty = int(((frame.cols-x)*vy/vx)+y);
                    cv::line(frame,cv::Point(frame.cols-1,righty),cv::Point(0,lefty),cv::Scalar(192,128,0));
                    
                    
               
                    drawContours(temp, contours,i, cvScalar(255), CV_FILLED, 8);
                }

                
                else if((blob_angle_deg <-10 || blob_angle_deg >10 ) && ((blob_angle_deg > -70 && blob_angle_deg < 70 ) || (rotated_rect.center.y > topLine && (rotated_rect.center.x > leftLine ||rotated_rect.center.x <rightLine))))
                {
                    
                    if ((contour_length/contour_width)>=ratio || (contour_width/contour_length)>=ratio ||(contour_area< smallLaneArea &&  ((contour_area/(contour_width*contour_length)) > .75) && ((contour_length/contour_width)>=3 || (contour_width/contour_length)>=3)))
                    {
                        drawContours(frame, contours,i, cvScalar(0), CV_FILLED, 8);
                        cv::Vec4f line;
                        cv::fitLine(contours[i], line, CV_DIST_L2, 0, 0.01, 0.01);
                        
                        float vx,vy,x,y;
                        vx = line[0];
                        vy = line[1];
                        x =line[2];
                        y=line[3];
                        int lefty,righty;
                        
                        lefty = int((-x*vy/vx) + y);
                        righty = int(((frame.cols-x)*vy/vx)+y);
                        cv::line(frame,cv::Point(frame.cols-1,righty),cv::Point(0,lefty),cv::Scalar(192,128,0));

                        drawContours(temp, contours,i, cvScalar(255), CV_FILLED, 8);
                    }
                }
            }
        }
    }
    
   // binaryImage.release();
    
    return temp;

};


'''


if __name__=="__main__":
	mtx,dist,mapx,mapy = calibrate(False)
	img = cv2.imread("../test_images/test1.jpg")
	color = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
	binary = getBinaryImageframe(color)
	cv2.imshow("binary",binary*255)
	cv2.waitKey()
	cv2.destroyAllWindows()
