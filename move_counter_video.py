import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

model_path = 'pose_landmarker_full.task'
file_name_image_model = "media/board-video3.png"
file_name_video = "media/video3.mp4"
options = python.vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=python.vision.RunningMode.VIDEO)

def GetCountoursFromImageModel(file_name_image_model):
    image_model = cv2.imread(file_name_image_model)
    gray = cv2.cvtColor(image_model, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)
    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)
    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw the contours on a copy of the original image
    cv2.drawContours(image_model, contours, -1, (0, 255, 0), 2)
    #print(len(contours), "objects were found in this image.")
    #cv2.imshow("Frame", image_model)
    return contours, image_model

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

def point_near_contour(point, contours, distance_limit=20):
    for contour in contours:
        for point_contour in contour:
            distance = calculate_distance_between_two_points(point, tuple(point_contour[0]))
            if distance < distance_limit:
                return True
    return False

def convert_normalized_to_pixels(point, width, height):
    return (int(point.x * width), int(point.y * height))

def main():
    counter_left_hand = 0
    counter_right_hand = 0
    counter_left_foot = 0
    counter_right_foot = 0

    # Get contours of te image model (without an human in the image)
    contours, image_model_contours = GetCountoursFromImageModel(file_name_image_model);
    #cv2.imshow("Frame", image_model_contours) #uncomment to see the image with contours
    #cv2.waitKey(0) #uncomment to see the cv2.imshow line above before continue, then press any key to continue
    #print(len(contours), "objects were found in this image.") #uncomment to see the len of image contours

    with python.vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Use OpenCV’s VideoCapture to load the input video.
        cap = cv2.VideoCapture(file_name_video)
        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        # You’ll need it to calculate the timestamp for each frame.
        frame_timestamp_ms = 0

        # Loop through each frame in the video using VideoCapture#read()
        while cap.isOpened():
            ret, frame_from_opencv = cap.read()
            if ret == True:
                # Resize image
                height_window = 600  # Set the height
                width_window = int(frame_from_opencv.shape[1] * (height_window / frame_from_opencv.shape[0]))
                frame_from_opencv = cv2.resize(frame_from_opencv, (width_window, height_window))

                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_from_opencv)
                frame_timestamp_ms = int(frame_timestamp_ms + 1000 / fps)
                # Perform pose landmarking on the provided single image.
                # The pose landmarker must be created with the video mode.
                pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                if pose_landmarker_result.pose_landmarks:
                    frame_from_opencv = draw_landmarks_on_image(frame_from_opencv, pose_landmarker_result)

                    # get width and height from image
                    height_image, width_image, _ = frame_from_opencv.shape
                    # Convert coordinates normalized from pose landmarks to pixels coordinates
                    points_body_pixels = [convert_normalized_to_pixels(point, width_image, height_image) for point in pose_landmarker_result.pose_landmarks[0]]
                    
                    point = points_body_pixels[19] #point 19 of body landmark locations
                    if point_near_contour(point, contours, 1):
                        #print("Landmark of a contour left hand index.")
                        counter_left_hand += 1 

                    point = points_body_pixels[20] #point 20 of body landmark locations
                    if point_near_contour(point, contours, 1):
                        #print("Landmark of a contour right hand index.")
                        counter_right_hand += 1

                    point = points_body_pixels[31] #point 31 of body landmark locations
                    if point_near_contour(point, contours, 1):
                        #print("Landmark of a contour left foot index.")
                        counter_left_foot += 1

                    point = points_body_pixels[32] #point 32 of body landmark locations
                    if point_near_contour(point, contours, 1):
                        #print("Landmark of a contour right foot index.")
                        counter_right_foot += 1

                    cv2.imshow("Frame", frame_from_opencv)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                break
        
        cap.release()
        cv2.destroyAllWindows()

        print(f"Moves on left hand: {counter_left_hand}")
        print(f"Moves on right hand: {counter_right_hand}")
        print(f"Moves on left foot: {counter_left_foot}")
        print(f"Moves on right foot: {counter_right_foot}")

if __name__ == "__main__":
    main()