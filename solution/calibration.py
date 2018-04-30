import numpy as np
import cv2
import glob
def calibrate( ShowResult = False):
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
	
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.
	
	# Make a list of calibration images
	images = glob.glob('../camera_cal/calibration*.jpg')
	testimg = cv2.imread(images[0])
	imshape = testimg.shape
	w=imshape[1]
	h=imshape[0]
	c=imshape[2]
	# Step through the list and search for chessboard corners
	for fname in images:
	    img = cv2.imread(fname)
	    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	    # Find the chessboard corners
	    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
	
	    # If found, add object points, image points
	    if ret == True:
	        objpoints.append(objp)
	        imgpoints.append(corners)
	
	        # Draw and display the corners
	        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
	#        cv2.imshow('img',img)
	
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
	print ("calibration result camera intrinsic matrix is {},\n distortion coefficients is{} ".format(mtx,dist))
	if ShowResult:
		for fname in images:
			img = cv2.imread(fname)
			#dst = cv2.undistort(img, mtx, dist, None, mtx)
			dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
			cv2.imshow("udist",dst)
			cv2.waitKey()
		cv2.destroyAllWindows()
	return [mtx,dist,mapx,mapy];


if __name__=="__main__":
	calibrate(True)

