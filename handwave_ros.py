import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge

# Hold the background frame for background subtraction.
background = None
# Hold the hand's data so all its details are in one place.
hand = None
# Variables to count how many frames have passed and to set the size of the window.
FRAME_HEIGHT = 200
FRAME_WIDTH = 300
# Humans come in a ton of beautiful shades and colors.
# Try editing these if your program has trouble recognizing your skin tone.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH
next_state = False
#region_bottom = FRAME_HEIGHT
#region_left = 0

class HandData(Node):
    top = (0,0)
    bottom = (0,0)
    left = (0,0)
    right = (0,0)
    centerX = 0
    prevCenterX = 0
    isInFrame = False
    isWaving = False
    fingers = None
    wave_count = 0
    wave_list = [0.0]

    
    def __init__(self, top, bottom, left, right, centerX):
        super().__init__("wave")
        self.get_logger().info("INIT")
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        isInFrame = False
        isWaving = False
        self.waving_pub = self.create_publisher(Bool, "/is_waving", 10)
        
    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def check_for_waving(self, centerX):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        print(self.wave_list)
        if abs(self.centerX - self.prevCenterX > 10):
            self.isWaving = True
            print("WAVE")
            self.get_logger().info("WAVE")
            self.wave_count += 1
            self.wave_list = np.append(self.wave_list,1.0)
        else:
            self.isWaving = False
            self.wave_count += 1
            self.wave_list = np.append(self.wave_list,0.0)
        if self.wave_count >= 3:
            if(np.mean(self.wave_list) > .1):
                print("NAV TO POI") #will replace with navigation procedure
                self.get_logger().info("ITWORKED")
                output=Bool()
                output.data = True
                self.waving_pub.publish(output)
                self.wave_count = 0
                self.wave_list = [0]
                raise SystemExit(0)
                return True
            else:
                self.wave_count = 0
                self.wave_list = [0]
                output=Bool()
                output.data = False
                self.waving_pub.publish(output)
                return False
        else:
            output=Bool()
            output.data = False
            self.waving_pub.publish(output)
            return False

# Here we take the current frame, the number of frames elapsed, and how many fingers we've detected
# so we can print on the screen which gesture is happening (or if the camera is calibrating).
def write_on_image(frame, frames_elapsed):
    text = "Searching..."
    #if frames_elapsed < CALIBRATION_TIME:
        #text = "Calibrating..."
    if hand == None or hand.isInFrame == False:
        text = "No hand detected"
    else:
        if hand.isWaving:
            text = "Waving"
        elif hand.fingers == 0:
            text = "Rock"
        elif hand.fingers == 1:
            text = "Pointing"
        elif hand.fingers == 2:
            text = "Scissors"
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)

    # Highlight the region of interest.
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)  
def get_region(frame):
    # Separate the region of interest from the rest of the frame.
    region = frame[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent frame noise from being labeled as an edge.
    region = cv2.GaussianBlur(region, (5,5), 0)

    return region
def get_average(region):
    # We have to use the global keyword because we want to edit the global variable.
    global background
    # If we haven't captured the background yet, make the current region the background.
    if background is None:
        background = region.copy().astype("float")
        return
    # Otherwise, add this captured frame to the average of the backgrounds.
    cv2.accumulateWeighted(region, background, BG_WEIGHT)
# Here we use differencing to separate the background from the object of interest.
def segment(region):
    global hand
    # Find the absolute difference between the background and the current frame.
    diff = cv2.absdiff(background.astype(np.uint8), region)

    # Threshold that region with a strict 0 or 1 ruling so only the foreground remains.
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # Get the contours of the region, which will return an outline of the hand.
    (contours, _) = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If we didn't get anything, there's no hand.
    if len(contours) == 0:
        if hand is not None:
            hand.isInFrame = False
        return
    # Otherwise return a tuple of the filled hand (thresholded_region), along with the outline (segmented_region).
    else:
        if hand is not None:
            hand.isInFrame = True
        segmented_region = max(contours, key = cv2.contourArea)
        return (thresholded_region, segmented_region)
def get_hand_data(thresholded_image, segmented_image, frames_elapsed):
    global hand
    
    # Enclose the area around the extremities in a convex hull to connect all outcroppings.
    convexHull = cv2.convexHull(segmented_image)
    
    # Find the extremities for the convex hull and store them as points.
    top    = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left   = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right  = tuple(convexHull[convexHull[:, :, 0].argmax()][0])
    
    # Get the center of the palm, so we can check for waving and find the fingers.
    centerX = int((left[0] + right[0]) / 2)
    
    # We put all the info into an object for handy extraction (get it? HANDy?)
    if hand == None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update(top, bottom, left, right)
    
    # Only check for waving every 6 frames.
    if frames_elapsed % 6 == 0:
        if(hand.check_for_waving(centerX)):
            next_state = False

def main(frame):

    frames_elapsed = 0

        
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
    frame = cv2.flip(frame, 1) 
    # Separate the region of interest and prep it for edge detection.
    region = get_region(frame)
    
    get_average(region)
    
    region_pair = segment(region)
    if region_pair is not None:
        # If we have the regions segmented successfully, show them in another window for the user.
        (thresholded_region, segmented_region) = region_pair
        cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
        #cv2.imshow("Segmented Image", region)
        
        get_hand_data(thresholded_region, segmented_region, frames_elapsed)

    # Write the action the hand is doing on the screen, and draw the region of interest.
    write_on_image(frame, frames_elapsed)
    # Show the previously captured frame.
    #cv2.imshow("Camera Input", frame)
    frames_elapsed += 1
    
        
    #if (cv2.waitKey(1) & 0xFF == ord('r')):
        #frames_elapsed = 0

    # When we exit the loop, we have to stop the capture too.
    #cv2.destroyAllWindows()

class RosImageSub(Node):
    def __init__(self):
        super().__init__("hand_wave")
        
        qos_profile = rclpy.qos.QoSProfile(
             reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
             history=rclpy.qos.HistoryPolicy.KEEP_LAST,
             depth=1
         )
        self.image_sub = self.create_subscription(CompressedImage, "/stereo/left/compressed", self.image_callback, qos_profile)
        #self.get_logger().info("INIT")
        self.cv_bridge = CvBridge()
    
    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            #self.get_logger().info("IMAGE")
            main(cv_image)

        except Exception as e:
            self.get_logger().error("Error Processing image: {}".format(str(e))) 

def ros_main():
    #rclpy.init()
    ros_sub = RosImageSub()
    try:
        rclpy.spin(ros_sub)
    except SystemExit as e:
        print("EXIT")
    ros_sub.destroy_node()
    #rclpy.shutdown()

if __name__=="__main__":
    ros_main()   