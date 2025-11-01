import socket
import json
import base64
import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 54321

# tăng buffer lên cho chắc
MAX_BUF = 500_000   # 500 KB

def connect_map():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print(f"[OK] Connected to map {HOST}:{PORT}")
    return s

def recv_frame(sock):
    """
    Nhận 1 gói JSON. Có debug.
    """
    try:
        data = sock.recv(MAX_BUF)
    except Exception as e:
        print("[ERR] recv error:", e)
        return None, None, None

    if not data:
        print("[WARN] empty data from simulator")
        return None, None, None

    # debug: xem gói to bao nhiêu
    print(f"[DBG] received {len(data)} bytes")

    try:
        pkg = json.loads(data)
    except Exception as e:
        print("[ERR] json.loads failed:", e)
        # in thử vài ký tự đầu để xem simulator gửi gì
        print(data[:200])
        return None, None, None

    # giải mã ảnh
    if "Img" not in pkg:
        print("[ERR] 'Img' key not found in JSON")
        return None, None, None

    try:
        img_bytes = base64.b64decode(pkg["Img"])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print("[ERR] image decode failed:", e)
        return None, None, None

    angle = pkg.get("Angle", 0.0)
    speed = pkg.get("Speed", 0.0)
    return frame, angle, speed

def send_control(sock, angle_cmd, speed_cmd):
    msg = f"{angle_cmd} {speed_cmd}"
    try:
        sock.sendall(msg.encode("utf-8"))
    except Exception as e:
        print("[ERR] send_control failed:", e)

if __name__ == "__main__":
    sock = connect_map()

    try:
        while True:
            frame, ang, spd = recv_frame(sock)
            if frame is None:
                # nếu lỗi thì tiếp tục chứ đừng thoát liền
                continue

            # resize cho chắc
            frame = cv2.resize(frame, (640, 480))

            # vẽ info
            cv2.putText(frame, f"sim angle: {ang}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"sim speed: {spd}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # HIỂN THỊ ĐÚNG TÊN EM MUỐN
            cv2.imshow("AGV View", frame)

            # gửi lệnh tạm (0 0)
            send_control(sock, 0, 0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        sock.close()
        # 
      
