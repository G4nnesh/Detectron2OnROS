#!/home/elmehdi_c/anaconda3/envs/Deep/bin/python

import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2 
 
def callback(data):
 
  br = cv_bridge.CvBridge()
 
  rospy.loginfo("receiving video frame")
   
  current_frame = br.imgmsg_to_cv2(data) #conversion par bridge
   
  cv2.imshow("camera", current_frame) #Affichage
   
  cv2.waitKey(1)
      
def receive_message():
 
  rospy.init_node('video_sub_py', anonymous=True)
   
  rospy.Subscriber('obstacles', Image, callback)
 
  rospy.spin()
 
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  receive_message()