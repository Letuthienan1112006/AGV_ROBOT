import socket
import json
import base64
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import road            
import record
from pid_control import PID

pid= PID(kp=0.08, ki=0.0, kd=0.02)

model= YOLO("yolov8s.pt")
last_alert_ts=0
alert_cooldown=8

# ================== CẤU HÌNH ==================
HOST = "127.0.0.1"     # simulator chạy trên máy 
PORT = 54321           # cổng simulator đang mở
MAX_BUF = 500_000      # đủ lớn để nhận ảnh base64
WIN_NAME = "AGV View"  # tên cửa sổ hiển thị
# =================================================


def connect_map(host=HOST, port=PORT):
    """Kết nối TCP tới simulator và trả về socket."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    
    s.settimeout(0.1)
    print(f"[OK] Connected to map {host}:{port}")
    return s


def recv_frame(sock):
    """
    Thử nhận 1 gói JSON từ simulator.
    Trả về: (frame_bgr, angle_from_sim, speed_from_sim)
    Nếu chưa có dữ liệu thì trả về (None, None, None)
    """
    try:
        data = sock.recv(MAX_BUF)
    except socket.timeout:
        
        return None, None, None

    if not data:
        return None, None, None

    try:
        pkg = json.loads(data)
    except Exception:
        
        return None, None, None

   
    try:
        img_bytes = base64.b64decode(pkg["Img"])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception:
        return None, None, None

    angle = pkg.get("Angle", 0.0)
    speed = pkg.get("Speed", 0.0)
    return frame, angle, speed


def send_control(sock, angle_cmd, speed_cmd):
    """
    Gửi lệnh điều khiển về simulator.
    Simulator của em đang nhận format: "angle speed"
    """
    msg = f"{angle_cmd} {speed_cmd}"
    try:
        sock.sendall(msg.encode("utf-8"))
    except Exception as e:
        print("[send_control ERR]", e)


if __name__ == "__main__":
    sock = connect_map()

    try:
        while True:
            
            frame, angle, speed = recv_frame(sock)
            if frame is None:
                continue
            frame = cv2.resize(frame, (640,480))

            
            off, mask = road.lane_offset(frame, roi_top=300, white=True)
            if off is None:
                
                steer = 0
            else:
                error = -off
                steer = pid.update(error)

            
            speed_cmd = 15  
            send_control(sock, steer, speed_cmd)








            
            send_control(sock, 0, 0)

            
            frame, sim_angle, sim_speed = recv_frame(sock)
            if frame is None:
                
                continue

            
            frame = cv2.resize(frame, (640, 480))
            #####
            off,mask=road.lane_offset(frame,roi_top=300,white=True)

            res=model(frame, imgsz=640, conf=0.6, verbose=False)[0]
            best_conf=0.0
            for b in res.boxes:
                if int(b.cls)==0:  # class 0=person
                    c=float(b.conf); best_conf=max(best_conf,c)
                    x1,y1,x2,y2=map(int,b.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame,f"person {c:.2f}",(x1,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            now=time.time()
            if best_conf>0.7 and now-last_alert_ts>alert_cooldown:
                cv2.imwrite('snap.jpg',frame)
                # threading.Thread(target=record.record_clip, args=(cap,6,"clip.mp4",20), daemon=True).start()
                print(f"[ALERT] Person detected conf={best_conf:.2f}")
                last_alert_ts=now

            cv2.putText(frame,f"offset:{0 if off is None else int(off)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.imshow("AGV View", frame)
            if cv2.waitKey(1)==ord('q'): break
                    



            

     
            angle_cmd = 0
            speed_cmd = 20
            # =================================================

            # gửi điều khiển thực
            send_control(sock, angle_cmd, speed_cmd)

            # ===== (C) DEBUG / HIỂN THỊ THÊM =====
            cv2.putText(frame, f"sim angle: {sim_angle}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"sim speed: {sim_speed}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"cmd: {angle_cmd:.1f} / {speed_cmd:.1f}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            # =====================================

            cv2.imshow(WIN_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("[DONE] disconnected")
