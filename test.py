import matplotlib.pyplot as plt
import numpy as np

# ======= PID class =======
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i, self.prev_e = 0, 0
    def step(self, e, dt):
        self.i += e*dt
        d = (e - self.prev_e)/dt
        self.prev_e = e
        return self.kp*e + self.ki*self.i + self.kd*d

# ======= mô phỏng xe =======
dt = 0.05
t = np.arange(0, 10, dt)
pid = PID(kp=0.03, ki=0.000, kd=0.15)

lane_center = 0
y = 0.8       # vị trí ban đầu lệch phải 0.8 m
yaw = 0
steer_log, y_log = [], []

for _ in t:
    error = lane_center - y
    steer = pid.step(error, dt)
    # mô hình động học rất đơn giản:
    # góc lái -> thay đổi hướng, cập nhật vị trí
    yaw += steer * dt
    y += np.sin(yaw) * 0.2   # xe tiến 0.2m mỗi bước

    steer_log.append(np.degrees(steer))
    y_log.append(y)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(t, [lane_center]*len(t),'--',label='Lane center')
plt.plot(t, y_log,label='Car Y position')
plt.xlabel('Time (s)'); plt.ylabel('Lateral position (m)')
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(t, steer_log,label='Steer (deg)')
plt.xlabel('Time (s)'); plt.ylabel('Steer angle (°)')
plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()
