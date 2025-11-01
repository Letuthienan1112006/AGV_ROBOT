# client_agv_full.py
# Kết nối simulator + lane + PID + YOLO + alert
# (dựa đúng code đồ án em đưa)

import socket
import json
import base64
import cv2
import numpy as np
import time
import threading

from ultralytics import YOLO
import road            # road.py của em
import record          # record.py của em (đã gửi)

# ================== CẤU HÌNH ==================
HOST = "127.0.0.1"
PORT = 54321
MAX_BUF = 500_000      # để đủ cho ảnh base64
RESIZE_W, RESIZE_H = 640, 480

# PID cho lái
KP = 0.08
KI = 0.0
KD = 0.02
ANGLE_LIMIT = 25       # simulator của em đang giới hạn
PIXEL_TO_DEG = 0.25    # chuyển pixel lệch vạch -> góc
DEFAULT_SPEED = 20     # chạy chậm để test
ALERT_COOLDOWN = 8     # giây
# ==============================================


class PID:
    def __init__(self, kp, ki, kd, out_min=-ANGLE_LIMIT, out_max=ANGLE_LIMIT):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0
        dt = now - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.prev_time = now

        # P, I, D
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        # giới hạn
        out = max(self.out_min, min(self.out_max, out))
        return out


def connect_map():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    # không chờ vô hạn
    s.settimeout(0.1)
    print(f"[OK] Connected to map {HOST}:{PORT}")
    return s


def recv_frame(sock):
    """Nhận 1 frame từ simulator. Nếu chưa có thì trả None."""
    try:
        data = sock.recv(MAX_BUF)
    except socket.timeout:
        return None, None, None
    except Exception as e:
        print("[ERR recv]", e)
        return None, None, None

    if not data:
        return None, None, None

    try:
        pkg = json.loads(data)
    except Exception as e:
        print("[ERR json]", e)
        return None, None, None

    # giải mã ảnh
    try:
        img_bytes = base64.b64decode(pkg["Img"])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print("[ERR decode img]", e)
        return None, None, None

    angle = pkg.get("Angle", 0.0)
    speed = pkg.get("Speed", 0.0)
    return frame, angle, speed


def send_control(sock, angle, speed):
    """Gửi điều khiển lại simulator."""
    msg = f"{angle} {speed}"
    try:
        sock.sendall(msg.encode("utf-8"))
    except Exception as e:
        print("[ERR send]", e)


if __name__ == "__main__":
    # 1. kết nối
    sock = connect_map()

    # 2. tạo PID
    pid_steer = PID(KP, KI, KD)

    # 3. load YOLO
    model = YOLO("yolov8s.pt")

    last_alert_ts = 0

    try:
        while True:
            # gửi keep-alive trước để simulator chịu gửi ảnh
            send_control(sock, 0, 0)

            # 4. nhận frame
            frame, sim_angle, sim_speed = recv_frame(sock)
            if frame is None:
                # chưa có ảnh thì lặp lại
                continue

            # resize giống code cũ của em
            frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

            # ============ 5. LANE (code đồ án của em) ============
            off, mask = road.lane_offset(frame, roi_top=300, white=True)

            # chuyển offset (pixel) -> error
            if off is None:
                # không thấy vạch → không PID
                steer_cmd = 0
                pid_steer.reset()
            else:
                # -off để đánh đúng chiều (nếu ngược thì bỏ dấu -)
                error = -off * PIXEL_TO_DEG
                steer_cmd = pid_steer.update(error)
            # ======================================================

            # ============ 6. YOLO (code đồ án của em) =============
            res = model(frame, imgsz=640, conf=0.6, verbose=False)[0]
            best_conf = 0.0
            for b in res.boxes:
                if int(b.cls) == 0:  # person
                    c = float(b.conf)
                    best_conf = max(best_conf, c)
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"person {c:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
            # ======================================================

            # ============ 7. ALERT + LƯU ẢNH ======================
            now = time.time()
            if best_conf > 0.7 and now - last_alert_ts > ALERT_COOLDOWN:
                cv2.imwrite("snap.jpg", frame)
                print(f"[ALERT] Person detected conf={best_conf:.2f}")

                # đây là chỗ code cũ của em: record từ camera
                # simulator không phải webcam nên có thể không record được
                try:
                    # em có thể đổi thành record từ webcam thật nếu muốn
                    threading.Thread(
                        target=record.record_clip,
                        args=(cv2.VideoCapture(0), 6, "clip.mp4", 20),
                        daemon=True,
                    ).start()
                except Exception as e:
                    print("[record] skip:", e)

                last_alert_ts = now
            # ======================================================

            # ============ 8. QUYẾT ĐỊNH TỐC ĐỘ ====================
            if best_conf > 0.7:
                speed_cmd = 0  # dừng khi có người
            else:
                speed_cmd = DEFAULT_SPEED
            # ======================================================

            # 9. gửi điều khiển thực sự
            send_control(sock, steer_cmd, speed_cmd)

            # 10. hiển thị
            cv2.putText(
                frame,
                f"offset: {0 if off is None else int(off)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"PID steer: {steer_cmd:.1f}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"sim angle: {sim_angle:.1f}  sim speed: {sim_speed:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # vẽ mask lane của em
            if mask is not None:
                h_roi, w_roi = mask.shape
                y0 = 300
                if y0 + h_roi <= frame.shape[0]:
                    roi = frame[y0:y0+h_roi, 0:w_roi, :]
                    roi[:, :, 2] = cv2.addWeighted(
                        roi[:, :, 2], 0.5, mask, 0.5, 0
                    )

            cv2.imshow("AGV View", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("[DONE] disconnected")
