import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow

model_path = 'pose_landmarker_full.task'
file_name = "video2.mp4"

options = python.vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=python.vision.RunningMode.VIDEO)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

with python.vision.PoseLandmarker.create_from_options(options) as landmarker:
            # Use OpenCV’s VideoCapture to load the input video.
            cap = cv2.VideoCapture(file_name)
            # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            # You’ll need it to calculate the timestamp for each frame.
            frame_timestamp_ms = 0

            # Loop through each frame in the video using VideoCapture#read()
            while cap.isOpened():
                ret, numpy_frame_from_opencv = cap.read()
                if ret == True:
                    # Resize image
                    height_window = 600  # Set the height
                    width_window = int(numpy_frame_from_opencv.shape[1] * (height_window / numpy_frame_from_opencv.shape[0]))  # Manter a proporção da imagem
                    numpy_frame_from_opencv = cv2.resize(numpy_frame_from_opencv, (width_window, height_window))

                    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                    frame_timestamp_ms = int(frame_timestamp_ms + 1000 / fps)
                    # Perform pose landmarking on the provided single image.
                    # The pose landmarker must be created with the video mode.
                    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                    
                    gray = cv2.cvtColor(numpy_frame_from_opencv, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                    edged = cv2.Canny(blurred, 10, 100)
                    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # draw the contours on a copy of the original image
                    cv2.drawContours(numpy_frame_from_opencv, contours, -1, (0, 255, 0), 2)

                    numpy_frame_from_opencv = draw_landmarks_on_image(numpy_frame_from_opencv, pose_landmarker_result)

                    cv2.imshow("Frame", numpy_frame_from_opencv)
                    #cv2_imshow(numpy_frame_from_opencv)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                
                else:
                      break
            
            cap.release()
            cv2.destroyAllWindows()