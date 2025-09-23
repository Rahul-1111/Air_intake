import os
import time
import cv2
from ultralytics import YOLO
from camera import VideoCamera
import pymcprotocol

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
    mc = None

# === PLC Functions ===
def read_trigger():
    if not plc_connected or mc is None:
        return True  # Auto-trigger in test mode
    
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

def write_result(result_value):
    """Write OK/NG result to PLC"""
    if not plc_connected or mc is None:
        result_text = "OK" if result_value == 1 else "NG"
        print(f"🧪 TEST MODE - Would send to PLC → D11: {result_value} ({result_text})")
        return
        
    try:
        mc.batchwrite_wordunits(headdevice="D11", values=[result_value])
        result_text = "OK" if result_value == 1 else "NG"
        print(f"📤 Sent Result → D11: {result_value} ({result_text})")
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
    
    # Initialize result as OK (1)
    detection_result = 1  # 1 = OK, 0 = NG
    detected_objects = []

    # Process detections
    if results.boxes is not None and len(results.boxes.data) > 0:
        for box in results.boxes.data:
            cls = int(box[-1])
            class_name = model.names[cls]
            confidence = float(box[4])
            
            detected_objects.append(class_name)
            
            # Draw bounding box
            xyxy = box[:4].cpu().numpy().astype(int)
            
            # Set color based on class
            if class_name.lower() == 'ok':
                color = (0, 255, 0)  # Green for OK
                detection_result = 1  # OK
            else:  # NG or any other class
                color = (0, 0, 255)  # Red for NG
                detection_result = 0  # NG
            
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", 
                       (xyxy[0], xyxy[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # No detections found
        detection_result = 0  # NG (no object detected)
        detected_objects = ["No objects detected"]

    # Determine final result
    if detected_objects:
        # Check if any NG detected
        ng_detected = any('ng' in obj.lower() for obj in detected_objects)
        ok_detected = any('ok' in obj.lower() for obj in detected_objects)
        
        if ng_detected:
            detection_result = 0  # NG
        elif ok_detected:
            detection_result = 1  # OK
        else:
            detection_result = 0  # NG (unknown objects)
    
    result_text = "✅ OK" if detection_result == 1 else "❌ NG"
    print(f"🧠 Detection Result: {result_text}")
    print(f"📋 Detected: {detected_objects}")
    
    # Send result to PLC
    write_result(detection_result)

    # Show frame for testing
    if not plc_connected:
        # Add result text on frame
        result_display = "OK" if detection_result == 1 else "NG"
        result_color = (0, 255, 0) if detection_result == 1 else (0, 0, 255)
        cv2.putText(frame, f"RESULT: {result_display}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3)
        
        cv2.imshow('OK/NG Detection System', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        elif key == ord(' '):  # Spacebar for manual trigger
            return "manual_trigger"
    
    return "success"

# === Main Loop ===
print("🚀 Starting OK/NG Detection System...")
print(f"📁 Using model: {model_path}")
print("🎯 Classes: OK (1) / NG (0)")

if plc_connected:
    print("🔗 PLC Mode: Waiting for trigger signal...")
else:
    print("🧪 Test Mode: Press SPACE for manual detection, Q to quit")

try:
    detection_count = 0
    ok_count = 0
    ng_count = 0
    
    while True:
        trigger_detected = False
        
        if plc_connected:
            # Normal PLC mode
            if read_trigger():
                trigger_detected = True
                print(f"🟢 PLC Trigger Received #{detection_count + 1}")
        else:
            # Test mode - auto trigger every 3 seconds
            trigger_detected = True
            print(f"🧪 Auto Detection #{detection_count + 1}")
        
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
                if detection_count >= 20:
                    print(f"🧪 Test completed after {detection_count} detections")
                    print(f"📊 Summary - OK: {ok_count}, NG: {ng_count}")
                    break
        
        # Sleep timing
        sleep_time = 0.2 if plc_connected else 3.0
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n⏹️ Stopped by user (Ctrl+C)")
    print(f"📊 Final Summary - Total: {detection_count}, OK: {ok_count}, NG: {ng_count}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
finally:
    if not plc_connected:
        cv2.destroyAllWindows()
    print("✅ Cleanup completed")