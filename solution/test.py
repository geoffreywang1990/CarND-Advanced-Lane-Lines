import numpy as np
import glob
import cv2
import calibration as calibrator
import findLines as detector


from moviepy.editor import VideoFileClip

mtx,dist,mapx,mapy = calibrator.calibrate(False)

def process_img(img,previous_l=[],previous_r = []):
	global mapx,mapy
	w = img.shape[1]
	h = img.shape[0]
	undistorted_image =  cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
	binary = detector.getBinaryImageframe(undistorted_image)
	
	M,invM = detector.compute_perspective_transform(undistorted_image.shape[1],undistorted_image.shape[0])
	perspective_binary = detector.apply_perspective_transform(binary,M)
	left_fit, right_fit, ploty, left_fitx, right_fitx,  leftx, lefty, rightx, righty = detector.identifyLines(perspective_binary,previous_l,previous_r)
	ret = detector.draw_lane(undistorted_image,perspective_binary,invM,left_fitx,right_fitx,ploty)
	
	
	curve = detector.compute_curvature(left_fit, right_fit, ploty, left_fitx, right_fitx, leftx, lefty, rightx, righty)
	offset = detector.compute_center_offset(curve, ret)
	ret = detector.render_curvature_and_offset(ret, curve, offset)
	return ret,left_fitx,right_fitx

def process_test_images():
	images = glob.glob('../test_images/*.jpg')

	for fname in images:
		img = cv2.imread(fname)
		ret,_,_= process_img(img)
		cv2.imwrite("../output_images/"+fname.split('/')[-1],ret)
		#cv2.imshow('find lane',ret)
		#cv2.waitKey()
def process_video(videoName):
#	video_input = VideoFileClip('../'+str(videoName)+'.mp4')
#	video_output = video_input.fl_image(process_img)
#	video_output.write_videofile('../'+str(videoName) + "_output.mp4", audio=False)
	cap = cv2.VideoCapture('../'+str(videoName)+'.mp4')
	ret,frame = cap.read()
	inputshape = frame.shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter('../'+str(videoName)+'_out.mp4',fourcc, 30.0, (inputshape[1],inputshape[0]))
	previous_l = []
	previous_r = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		try:
			result,previous_l,previous_r = process_img(frame,previous_l,previous_r)
		except:
			break
		cv2.imshow('find lane',result)
		#print(result.shape,inputshape)
		#cv2.waitKey()
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break
		out.write(result)
	
	cap.release()
	out.release()
	cv2.destroyAllWindows()

process_test_images()
process_video('project_video')
process_video('challenge_video')
process_video('harder_challenge_video')
