import cv2

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)  # ← Use index 1 for external camera
        if self.cap.isOpened():
            print("📷 External camera (index 1) initialized")
        else:
            print("❌ Failed to initialize external camera")

    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
