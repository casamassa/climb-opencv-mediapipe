import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

model_path = 'pose_landmarker_full.task'
file_name = "video3.mp4"

options = python.vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=python.vision.RunningMode.VIDEO)

previous_point = None
distance_total = 0

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

def calculate_distance_between_two_points(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

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

            if pose_landmarker_result.pose_landmarks:
                # Get the coordinates of a specific landmark (i.e., left wrist)
                current_point = (pose_landmarker_result.pose_landmarks[0][mp.solutions.pose.PoseLandmark.NOSE].x * numpy_frame_from_opencv.shape[1],
                            pose_landmarker_result.pose_landmarks[0][mp.solutions.pose.PoseLandmark.NOSE].y * numpy_frame_from_opencv.shape[0])
                # If there is previous point, calculate the distance between the points
                if previous_point is not None:
                    distance_frame = calculate_distance_between_two_points(previous_point, current_point)
                    distance_total += distance_frame
                    print(distance_total)
                # Update previous point as next point
                previous_point = current_point

                numpy_frame_from_opencv = draw_landmarks_on_image(numpy_frame_from_opencv, pose_landmarker_result)

                cv2.imshow("Frame", numpy_frame_from_opencv)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        
        else:
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Distancia em pixel: " + str(distance_total))
    print("Distancia em cm: " + str(distance_total * 0.85416667))
    print("Distancia em m: " + str((distance_total * 0.85416667)/100))
    