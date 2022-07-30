#!/usr/bin/env python

"""
ON THE RASPI: roslaunch raspicam_node camerav2_640x480.launch enable_raw:=true

   0------------------> x (cols) Image Frame
   |
   |        c    Camera frame
   |         o---> x
   |         |
   |         V y
   |
   V y (rows)


SUBSCRIBES TO:
    /raspicam_node/image: Source image topic
    
PUBLISHES TO:
    /lane/image_lane : image with detected lane


"""


#--- Allow relative importing
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    
import sys
import rospy
import cv2
import numpy as np
import time

from std_msgs.msg           import String
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError

Ts=0.142857

#choose to apply adaptive treshold filter
adaptive_filter = True

#Calibration Matrixes using OpenCV
camera_matrix = np.array([[309.5954735138005276, 0, 333.6373845671422487], [0, 309.9124724576500398, 218.6567383907390933], [0, 0, 1]])
camera_distortion = np.array([-0.3229882105859484542, 0.1147174227731293239, -0.0002659899101280842294, -0.0003059580333839108162, -0.01926201441698963818])

#Perspective transformation Points
src = np.float32([
    (389,205), 
    (283,205),
    (11,478),
    (645,478)
    ])
dst = np.float32([
        (520, 0),
        (200,0),
        (200, 661),
        (520, 661)
    ])


class LaneDetection:

    def __init__(self,perspective_matrix=None, perspective_matrix_inv=None):
        
        self.perspective_matrix = perspective_matrix
        self.perspective_matrix_inv = perspective_matrix_inv
        self.lane_info = Point()
    
            
        print (">> Publishing image to topic image_lane")
        self.hls_pub = rospy.Publisher("/lane/hls_lane",Image,queue_size=1)
        
        print (">> Publishing offset and angle for control_node")
        self.lane_pub  = rospy.Publisher("/lane_detection",Point,queue_size=1)
        
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/raspicam_node/image",Image,self.callback)
        print ("<< Subscribed to topic /raspicam_node/image")
      
         
    def callback(self,data):
    	#Ts=0.142857
     	#rate = rospy.Rate(1/Ts)
        #--- Assuming image is 640x480
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        (rows,cols,channels) = cv_image.shape
        #1 - Apply camera calibration data to undistort the image
        cv_image_undist   = cv2.undistort(cv_image, camera_matrix, camera_distortion, None, camera_matrix) 
        
        #2 - Apply perspective tranformation to have a bird's-eye view
        cv_image_undist   = cv2.warpPerspective(cv_image_undist, self.perspective_matrix, (720, 661), flags=cv2.INTER_LINEAR)
        
        #3 - Detecting the lanes with HLS colorspace and getting the binary mask image
        hls_image = cv2.cvtColor(cv_image_undist, cv2.COLOR_BGR2HLS)
        if adaptive_filter:
        	##Using Lightness channel to detect the lane and creating adaptive threshold
        	L = hls_image[:,:,1]
        	L_max, L_mean = np.max(L), np.mean(L)
        	##Tunning parameters for the adaptive threshold
        	L_adapt_white = max(30, int(L_max *0.7),int(L_mean * 1.25))
        	hls_low_white = np.array((0, L_adapt_white,  0))
        else:
        	hls_low_white = np.array((0, 80,  0)) 
        #print(hls_low_white)
        hls_high_white = np.array((255, 255, 255))
        hls_output = np.zeros_like(hls_image[:,:,0])
        mask = (hls_image[:,:,0] >= hls_low_white[0]) & (hls_image[:,:,0] <= hls_high_white[0]) & (hls_image[:,:,1] >= hls_low_white[1]) & (hls_image[:,:,1] <= hls_high_white[1]) & (hls_image[:,:,2] >= hls_low_white[2]) & (hls_image[:,:,2] <= hls_high_white[2])
        hls_output[mask] = 255 
        #4 - Fitting a curve and drawing the sliding windows
        x_mppx = 0.001395266;
        y_mppx = 0.001792525;
        offset, line_angle = polyfit_lane_detection(y_mppx, x_mppx, hls_output); 
        #Publish Image
        try:
        	self.hls_pub.publish(self.bridge.cv2_to_imgmsg(hls_output, "8UC1"))
        except CvBridgeError as e:
        	print(e)
        
        self.lane_info.x = offset
        self.lane_info.y = -line_angle   
        #self.lane_info.z = L_adapt_white
        self.lane_pub.publish(self.lane_info) 
        #rate.sleep()
            
def polyfit_lane_detection(y_mppx, x_mppx, binary):
    '''
    Detect lane lines in a thresholded binary image using the sliding window technique
    :param binary (ndarray): Thresholded binary image
    '''
    IMG_SHAPE = (720, 661)
    size, ysize = IMG_SHAPE
    
    #Find coordenates of nonzero points
    nonzero = binary.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    
    #Fit a 2nd order polynomial for each lane line pixels
    #Find Angle
    
    fit_real = np.polyfit(nonzeroy*y_mppx, nonzerox*x_mppx, 1)
    line_angle = np.arctan(fit_real[0]);
    
    #Find Offset
    fit_img = np.polyfit(nonzeroy, nonzerox, 1)
 
    # Get the points for the entire height of the image
    plot_y = np.linspace(0, ysize-1, ysize)
    plot_x = fit_img[0] * plot_y + fit_img[1]
    
    lane_center = plot_x[-1]
    car_center = IMG_SHAPE[0] / 2

    offset = (lane_center - car_center) * x_mppx
    
    return offset, line_angle
    
def main(args):
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    perspective_matrix_inv = cv2.getPerspectiveTransform(dst, src)
    
    rospy.init_node('lane_detection', anonymous=True)
    ic = LaneDetection(perspective_matrix, perspective_matrix_inv)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
