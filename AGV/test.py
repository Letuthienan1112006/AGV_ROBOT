import socket
import json
import base64
import cv2
import numpy as np
import time
from ultralytics import YOLO
import road
from pid_control import PID

pid = PID(kp=0.08, ki=0.0, kd=0.02)
model = YOLO("yolov8s.pt")
last_alert_ts = 0
alert_cooldown = 8

# ================== CẤU HÌNH ==================
HOST = "127.0.0.1"
PORT = 54321
MAX_BUF = 500_000
WIN_NAME = "AGV View"
# ==============================================


def connect_map(host=HOST, port=PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.settimeout(0.1)
    print(f"[OK] Connected to map {host}:{port}")
    return s


def recv_frame(sock):
    try:
        data = sock.recv(MAX_BUF)
    except socket.timeout:
        return None, None, None

    if not data:
        return None, None, None

    try:
        pkg = json.loads(data)
        img_bytes = base64.b64decode(pkg["Img"])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        angle = pkg.get("Angle", 0.0)
        speed = pkg.get("Speed", 0.0)
        return frame, angle, speed
    except Exception:
        return None, None, None


def send_control(sock, angle_cmd, speed_cmd):
    msg = f"{angle_cmd} {speed_cmd}"
    try:
        sock.sendall(msg.encode("utf-8"))
    except Exception as e:
        print("[send_control ERR]", e)


def process_frame(frame):
    """Xử lý frame: tính offset và phát hiện đối tượng."""
    try:
        off, mask = road.lane_offset(frame, roi_top=300, white=True)
    except AttributeError:
        off, mask = None, None

    res = model(frame, imgsz=640, conf=0.6, verbose=False)[0]
    best_conf = 0.0
    for b in res.boxes:
        if int(b.cls) == 0:  # class 0=person
            c = float(b.conf)
            best_conf = max(best_conf, c)
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"person {c:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return off, best_conf


if __name__ == "__main__":
    sock = connect_map()

    try:
        while True:
            frame, angle, speed = recv_frame(sock)
            if frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            off, best_conf = process_frame(frame)

            if off is None:
                steer = 0
            else:
                error = -off
                steer = pid.update(error)

            if best_conf > 0.7 and time.time() - last_alert_ts > alert_cooldown:
                cv2.imwrite('snap.jpg', frame)
                print(f"[ALERT] Person detected conf={best_conf:.2f}")
                last_alert_ts = time.time()

            send_control(sock, steer, 15)

            cv2.putText(frame, f"offset: {0 if off is None else int(off)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WIN_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("[DONE] disconnected")