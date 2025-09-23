import cv2

def test_cameras():
    print("🔍 Testing camera indices 0 to 4...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"❌ Camera index {i} not available")
            continue

        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera working at index {i}")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
        else:
            print(f"⚠️ Camera index {i} opened but no frame")
        cap.release()

if __name__ == "__main__":
    test_cameras()
