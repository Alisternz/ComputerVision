import cv2
import numpy as np

# Modifiable parameters
VIDEO_FILE = 'eyes.mp4'
MIN_RADIUS = 50 
MAX_RADIUS = 300
INTENSITY_THRESHOLD = 20000
CIRCULARITY_THRESHOLD = 0.97



#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class Circle:

    instanceCount = 0
    maxX = 0
    maxY = 0

    def __init__(self, x, y, radius, intensity, circularity, prev_x=None, prev_y=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.intensity = intensity
        self.circularity = circularity
        self.detected_frames = 0  # Counter to track how many frames the eye has been detected
        # Set previous positions to the provided values or to current position if none provided
        self.prevX = prev_x if prev_x is not None else x
        self.prevY = prev_y if prev_y is not None else y
        Circle.instanceCount += 1



    @classmethod
    def from_circle_detection(cls, detection_result, eye_region, prev_x=None, prev_y=None):
        x, y, radius = detection_result[0], detection_result[1], detection_result[2]
        avg_intensity = cls.circle_average_intensity(eye_region, (x, y, radius))
        circ_index = cls.circularity_index(eye_region, (x, y, radius))
        return cls(x, y, radius, avg_intensity, circ_index, prev_x, prev_y)


    def update_position(self, x, y, maxX, maxY):
        self.x = x
        self.y = y
        self.detected_frames += 1
        if ((self.x - self.prevX) > maxX):
            maxX = self.x - self.prevX
        if ((self.y - self.prevY) > maxY):
            maxY = self.y - self.prevY
        print(self.y - self.prevY)
        return maxX, maxY

    def process(self, kalman_filter, offset_x, frame, maxX, maxY):
        if self.intensity < INTENSITY_THRESHOLD and self.circularity > CIRCULARITY_THRESHOLD:
            kalman_prediction = kalman_filter.predict()
            predicted_x, predicted_y = kalman_prediction[0], kalman_prediction[1]
            kalman_corrected = kalman_filter.correct(np.array([[self.x], [self.y]], dtype=np.float32))
            corrected_x, corrected_y = kalman_corrected[0], kalman_corrected[1]
            
            # Draw circle on frame
            #cv2.circle(frame, (int(corrected_x) + offset_x, int(corrected_y)), self.radius, (0, 255, 0), 2)
            cv2.circle(frame, (int(corrected_x) + offset_x, int(corrected_y)), 10, (0, 0, 255), 3) #outside ring
            cv2.circle(frame, (int(corrected_x) + offset_x, int(corrected_y)), 4, (0, 255, 0), 3)  #indside ring


            cv2.putText(frame, f"{self.x:d}", (int(corrected_x) + offset_x - 50, int(corrected_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"{self.y:d}", (int(corrected_x) + offset_x - 20, int(corrected_y) +40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            
            cv2.putText(frame, f"X-Movement: {maxX}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 100), 2, cv2.LINE_AA) 
            cv2.putText(frame, f"Y-Movement: {maxY}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 100), 2, cv2.LINE_AA) 
        
                   

    @staticmethod
    def circle_average_intensity(image, circle):
        x, y, radius = circle
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, thickness=-1)
        masked_pixels = cv2.bitwise_and(image, image, mask=mask)
        mean_intensity = cv2.mean(masked_pixels, mask=mask)[0]  # cv2.mean returns a tuple
        return mean_intensity

    @staticmethod
    def circularity_index(image, circle):
        x, y, radius = circle
        contour = np.array([[(x + np.cos(theta) * radius, y + np.sin(theta) * radius)]
                            for theta in np.linspace(0, 2*np.pi, 100)], dtype=np.int32)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity

    @staticmethod
    def process_eye(eye_region, kalman_filter, detected_circles, maxX, maxY):
        circles = cv2.HoughCircles(eye_region, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                param1=50, param2=30, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
        last_circle = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                circle = (i[0], i[1], i[2])
                avg_intensity = Circle.circle_average_intensity(eye_region, circle)
                if avg_intensity < INTENSITY_THRESHOLD:
                    circ_index = Circle.circularity_index(eye_region, circle)
                    if circ_index > CIRCULARITY_THRESHOLD:
                        found = False
                        for detected_circle in detected_circles:
                            if abs(detected_circle.x - circle[0]) < 50 and abs(detected_circle.y - circle[1]) < 20:
                                detected_circle.update_position(circle[0], circle[1], maxX, maxY)
                                last_circle = detected_circle
                                found = True
                                break
                        if not found:
                            # If no similar circle found, create a new Circle object with optional previous positions
                            if last_circle:
                                new_circle = Circle.from_circle_detection(circle, eye_region, last_circle.x, last_circle.y)
                            else:
                                new_circle = Circle.from_circle_detection(circle, eye_region)
                            detected_circles.append(new_circle)
                            last_circle = new_circle
                        return circle
        return None





def initialize_kalman_filter():
    dt = 1.0  # Time step
    kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
    kalman.transitionMatrix = np.array([[1, 0, dt, 0],
                                         [0, 1, 0, dt],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], dtype=np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
    kalman.measurementNoiseCov = 10 * np.eye(2, dtype=np.float32)
    kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
    kalman.statePost = np.zeros((4, 1), dtype=np.float32)
    return kalman

def detect_eyes(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)) #Cascade Approach
    # for (x, y, w, h) in eyes:
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    processed_frame = cv2.medianBlur(gray_frame, 75)
    height, width = gray_frame.shape
    left_half = processed_frame[:, :width // 2]
    right_half = processed_frame[:, width // 2:]

    return left_half, right_half



def main():

    maxX = 0
    maxY = 0
    webcam = cv2.VideoCapture(VIDEO_FILE)
    kalman_left = initialize_kalman_filter()
    kalman_right = initialize_kalman_filter()
    detected_left_circles = []
    detected_right_circles = []

    while True:
        _, frame = webcam.read()
        if frame is None:
            break

        left_eye, right_eye = detect_eyes(frame)
        
        # Process left eye
        left_circle = Circle.process_eye(left_eye, kalman_left, detected_left_circles, maxX, maxY)
        if left_circle is not None:
            left_circle = Circle.from_circle_detection(left_circle, left_eye)
            left_circle.process(kalman_left, 0, frame, maxX, maxY)
        
        # Process right eye
        right_circle = Circle.process_eye(right_eye, kalman_right, detected_right_circles, maxX, maxY)
        if right_circle is not None:
            right_circle = Circle.from_circle_detection(right_circle, right_eye)
            right_circle.process(kalman_right, left_eye.shape[1], frame, maxX, maxY)

        cv2.imshow("Demo", frame)
        print(f"Total Circle instances created: {Circle.instanceCount}")

        if cv2.waitKey(1) == 27:  # ESC key to break
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
