import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# thông số xe
L = 0.25          # chiều dài cơ sở (m)
v = 0.3           # vận tốc (m/s)
dt = 0.05

# đường cần bám (hình sin hoặc thẳng)
path_x = np.arange(0, 8, 0.1)
path_y = 0.5 * np.sin(0.5 * path_x)

# trạng thái xe ban đầu
x, y, yaw = 0.0, -0.5, 0.0     # lệch dưới đường
L_d = 0.6                      # look-ahead distance

def pure_pursuit_control(x, y, yaw):
    # tìm điểm đích look-ahead
    dists = np.hypot(path_x - x, path_y - y)
    idx = np.argmin(np.abs(dists - L_d))
    tx, ty = path_x[idx], path_y[idx]

    # góc tới mục tiêu
    alpha = np.arctan2(ty - y, tx - x) - yaw
    # góc lái
    delta = np.arctan2(2*L*np.sin(alpha), L_d)
    return delta, tx, ty

# lưu dữ liệu để vẽ
xs, ys, yaws, deltas = [x], [y], [yaw], []

for _ in range(300):
    delta, tx, ty = pure_pursuit_control(x, y, yaw)
    # giới hạn góc lái ±30°
    delta = np.clip(delta, -np.radians(30), np.radians(30))
    # cập nhật trạng thái xe
    x += v*np.cos(yaw)*dt
    y += v*np.sin(yaw)*dt
    yaw += (v/L)*np.tan(delta)*dt

    xs.append(x); ys.append(y); yaws.append(yaw); deltas.append(delta)

# ---- animation ----
fig, ax = plt.subplots()
ax.plot(path_x, path_y, 'b--', label='Target path')
car, = ax.plot([], [], 'r-', lw=2, label='Car body')
look, = ax.plot([], [], 'go', label='Look-ahead')
ax.set_xlim(-1, 8); ax.set_ylim(-2, 2)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(); ax.grid(True)

def update(i):
    # hình chữ nhật mô tả xe
    car_len, car_wid = 0.3, 0.15
    cx, cy, th = xs[i], ys[i], yaws[i]
    car_x = [cx + car_len*np.cos(th), cx - car_len*np.cos(th)]
    car_y = [cy + car_len*np.sin(th), cy - car_len*np.sin(th)]
    car.set_data(car_x, car_y)
    # điểm look-ahead
    delta, tx, ty = pure_pursuit_control(cx, cy, th)
    look.set_data([tx],[ty])
    return car, look

ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=50)
plt.show()
