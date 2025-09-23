import os
import time
import cv2
from ultralytics import YOLO
from camera import VideoCamera
import pymcprotocol
import struct

# === PLC Configuration ===
PLC_IP = "192.168.1.12"
TRIGGER_COIL = "M10"   # Bit coil
RESULT_ADDR = 11       # D11

# === Load Model ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'best.pt')
model = YOLO(model_path)

# === Setup Camera ===
camera = VideoCamera()
print("📷 Camera initialized")

# === Setup PLC with Error Handling ===
mc = None
plc_connected = False

try:
    mc = pymcprotocol.Type3E()
    mc.connect(PLC_IP, 5007)
    plc_connected = True
    print(f"✅ Connected to PLC at {PLC_IP}")
except Exception as e:
    print(f"⚠️ PLC Connection Failed: {e}")
    print("🧪 Running in TEST MODE without PLC")
    print("📋 Detection will run automatically for testing")
    mc = None

# === PLC Bit Functions ===
def read_trigger():
    if not plc_connected or mc is None:
        # For testing without PLC - auto trigger every few seconds
        return True
    
    try:
        data = mc.batchread_bitunits(headdevice=TRIGGER_COIL, readsize=1)
        return data[0] == 1
    except Exception as e:
        print(f"❌ Trigger read error: {e}")
        return False

def reset_trigger():
    if not plc_connected or mc is None:
        return
        
    try:
        mc.batchwrite_bitunits(headdevice=TRIGGER_COIL, values=[0])
        print("🔄 PLC Trigger Reset (M10=0)")
    except Exception as e:
        print(f"❌ Trigger reset failed: {e}")

def write_result(val_d11, val_d21):
    if not plc_connected or mc is None:
        print(f"🧪 TEST MODE - Would send to PLC → D11: {val_d11}, D21: {val_d21}")
        return
        
    try:
        mc.batchwrite_wordunits(headdevice="D11", values=[val_d11])
        mc.batchwrite_wordunits(headdevice="D21", values=[val_d21])
        print(f"📤 Sent Results → D11: {val_d11}, D21: {val_d21}")
    except Exception as e:
        print(f"❌ PLC write error: {e}")

# === Detection Logic ===
def detect_once():
    frame = camera.get_frame()
    if frame is None:
        print("⚠️ No frame captured")
        return "no_frame"

    # Run YOLO detection
    results = model(frame, imgsz=640)[0]
    missing_d11 = 0  # For caps 1,3,5 → D11
    missing_d21 = 0  # For caps 2,4,6 → D21
    missing_caps = set()

    # Process detections
    for box in results.boxes.data:
        cls = int(box[-1])
        cap = cls + 1
        missing_caps.add(cap)

        # Draw bounding box
        xyxy = box[:4].cpu().numpy().astype(int)
        label = model.names[cls]
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode missing caps to PLC values
        if cap in [1, 3, 5]:
            bit = [1, 3, 5].index(cap)  # bit 0,1,2
            missing_d11 |= (1 << bit)
        elif cap in [2, 4, 6]:
            bit = [2, 4, 6].index(cap)  # bit 0,1,2
            missing_d21 |= (1 << bit)

    print(f"🧠 Missing caps: {sorted(missing_caps)} → D11: {missing_d11}, D21: {missing_d21}")
    write_result(missing_d11, missing_d21)

    # Show frame for testing (remove in production)
    if not plc_connected:
        cv2.imshow('YOLO Detection - Test Mode', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        elif key == ord(' '):  # Spacebar to manually trigger
            return "manual_trigger"
    
    return "success"

# === Main Loop ===
print("🚀 Starting cap detection system...")
print(f"📁 Using model: {model_path}")

if plc_connected:
    print("🔗 PLC Mode: Waiting for trigger signal...")
else:
    print("🧪 Test Mode: Press SPACE for manual detection, Q to quit")

try:
    detection_count = 0
    
    while True:
        trigger_detected = False
        
        if plc_connected:
            # Normal PLC mode
            if read_trigger():
                trigger_detected = True
                print(f"🟢 PLC Trigger Received #{detection_count + 1}")
        else:
            # Test mode - auto trigger every 3 seconds or manual trigger
            trigger_detected = True
            print(f"🧪 Auto Detection #{detection_count + 1} (Press SPACE for manual)")
        
        if trigger_detected:
            result = detect_once()
            
            if result == "quit":
                print("👋 Exiting...")
                break
            elif result == "no_frame":
                print("⚠️ Camera issue, retrying...")
            else:
                detection_count += 1
                
            if plc_connected:
                reset_trigger()
            else:
                # In test mode, limit to prevent spam
                if detection_count >= 10:
                    print("🧪 Test completed after 10 detections")
                    print("📋 To run with PLC, fix connection to 192.168.1.12:5007")
                    break
        
        # Sleep timing
        sleep_time = 0.2 if plc_connected else 3.0
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n⏹️ Stopped by user (Ctrl+C)")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
finally:
    if not plc_connected:
        cv2.destroyAllWindows()
    print("✅ Cleanup completed")