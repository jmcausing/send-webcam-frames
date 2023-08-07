import face_recognition
import cv2
import numpy as np
from datetime import datetime
import time
import requests
import sys, os, logging
import atexit
import io
import threading

class Video_face_scan:
    """
    Video Face detection that alerts if it detects an unknown face/intruder 
    """

    def __init__(self):
        # Setup default variables
        # We use this to process every other frame of video to save time
        self.process_this_frame = True
        # URL for cloud native server (Kinsta app to process facial recognition)
        self.cloud_native_app_url = "http://localhost:8080/accept_frames"

        # frame send limit and seconds pause
        # this will send 10 frame using threading and pause for 5 seconds then continue
        self.frame_counter = 0
        self.frame_counter_limit = 5
        self.frame_counter_limit_plus_one = 6
        self.frame_pause_seconds = 10

    def setup(self):
        # Checking if webcam exist..
        print('# Searching for camera..')
        for cam in range(-2,20):
            try:
                # For debug
                print(f'# Trying camera {cam}..')
                video_capture = cv2.VideoCapture(cam)
                frame_test = video_capture.read()
                if 'True' in str(frame_test):
                    print(f'# Found camera index: {cam}')
                    self.camera = cam
                    video_capture.release()

                    break
            except Exception as e:
                    print(e)
        if 'False' in str(frame_test):
            print('# Webcam not found!')   
            self.alert_and_shutdown(exitCode=1, msg='Setup() - Camera not found! Shutting down!')

        ## Set up logging
        self.log_path = 'logs/vfs.log'
        self.log_level = logging.INFO  ## possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.log_format = '%(asctime)s - [%(levelname)s]: %(message)s'
        self.log_date_format = '%Y-%m-%d %H:%M:%S'
        ls = self.logSetup(self.log_path, self.log_level, self.log_format, self.log_date_format)
        if ls != True:
            self.alert_and_shutdown(exitCode=1, msg='Setup() - Error setting up logs')
        print('# Setup finished!')

            
    # Shutdown exit function
    def alert_and_shutdown(self, msg=None, exitCode=0):
        # This helps for shutting down the process and alerting if needed.
        if msg is None:
            sys.exit(exitCode)
        else:
            # Send slack!!!
            print(f'# Shutting down! Error: {msg}')
            sys.exit(exitCode)

    # Function for logging setup
    def logSetup(self, log_path, log_level, log_format, log_date_format):
        # Setting up log facility
        target_dir = 'logs'
        if not os.path.isdir(target_dir):
            try:
                os.mkdir(target_dir)
            except OSError as e:
                logging.error(f"Couldn't create log folder: {e!r}\nShutting down.")
                return f'{e!r}'
        try:
            logging.basicConfig(filename=log_path, level=log_level, format=log_format, datefmt=log_date_format)
        except Exception as e:
            return f'{e!r}'
        else:
            logging.info('Logging setup complete')
            return True

    def send_frame(self, frame):
        buffer = io.BytesIO()
        np.save(buffer, frame)
        buffer.seek(0)

        files = {"frames": ("frame.npy", buffer, "application/octet-stream")}
        response = requests.post(self.cloud_native_app_url, files=files)

        # Check the response from server1
        if response.status_code == 200:
            print("Request successful!")
            print("Response:", response.text)
        else:
            print("Request failed:", response.status_code)
            print("Response:", response.text)

    # Function to start detecting faces
    def video_detect_start(self):
        # Get time and date
        now = datetime.now()        
        
        # Start video capture from webcam
        logging.info('video_detect_start() - Starting video_capture..')
        video_capture = cv2.VideoCapture(self.camera)
        # Infinite loop video frame capture starts here
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # Only process every other frame of video to save time
            if self.process_this_frame:

                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Get face location from current video frame
                face_locations = face_recognition.face_locations(small_frame)

                # If face found from current video frame, send it to the cloud native app 
                if bool(face_locations):

                    # Get time and date so we can log it!
                    now = datetime.now()
                    current_time = now.strftime("%m-%d-%Y--%I-%M-%S-%p")

                    # Send to logs
                    logging.info(f'Face detected! Sending to cloud native server {self.cloud_native_app_url}')
                    print(f'# Face detected! Sending to cloud native server {self.cloud_native_app_url}')

                    # Check frame counter and pause if necessary
                    if self.frame_counter == self.frame_counter_limit:
                        print(f"Pausing for {self.frame_pause_seconds} seconds...")
                        time.sleep(self.frame_pause_seconds)
                        self.frame_counter = 0  # Reset frame counter after pausing

                    # Create a thread to send the frame
                    thread = threading.Thread(target=self.send_frame, args=(frame,))
                    thread.start()

                    self.frame_counter += 1 

                    if self.frame_counter >= self.frame_counter_limit_plus_one:
                        self.frame_counter = 0  # Reset frame counter after reaching 10 frames

                ## For debug
                else: 
                   print(f'# No face found!! face_location: {face_locations}')

            self.process_this_frame = not self.process_this_frame
        # Release handle to the webcam
        video_capture.release()


    def run(self):
        print("# Starting video face detection...")
        # Setup our environment
        self.setup()
        # Run the main function video_detect_start()    
        start = self.video_detect_start()    
        return start
        
def detect_exit():
    logging.info(f'detect_exit() - Script was shutdown/interrupted')

def main(request): # 
    # Detect and log if the script was shutdown/exit
    atexit.register(detect_exit)
    check = Video_face_scan()
    return check.run()

if __name__ == "__main__": 
    try:
        main(None)
    except Exception as e:
        print(e)
