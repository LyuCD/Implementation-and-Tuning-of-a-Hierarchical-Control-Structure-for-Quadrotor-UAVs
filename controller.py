import numpy as np

# PID类保持不变
class PID:
    def __init__(self, kp, ki, kd, limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.limit is not None:
            output = max(min(output, self.limit), -self.limit)
        return output

# 初始化各层PID控制器
try:
    _controller_initialized
except NameError:
    _controller_initialized = True

    # 第一层：位置 PID
    pid_pos_x = PID(2.0, 0.0, 0.3, limit=2.0)
    pid_pos_y = PID(2.0, 0.0, 0.3, limit=2.0)
    pid_pos_z = PID(2.0, 0.0, 0.2, limit=2.0)

    # 第二层：速度 PID
    pid_vel_x = PID(2.0, 0.0, 0.2, limit=2.0)
    pid_vel_y = PID(2.0, 0.0, 0.2, limit=2.0)
    pid_vel_z = PID(2.0, 0.0, 0.2, limit=1.0)

    # 第三层：推力/加速度 PID（用于模拟电机控制）
    pid_thrust_x = PID(1.0, 0.0, 0.1, limit=1)  # 可设置为最大PWM值
    pid_thrust_y = PID(1.0, 0.0, 0.1, limit=1)
    pid_thrust_z = PID(2.0, 0.0, 0.5, limit=1)

    # 航向角 PID
    pid_yaw = PID(2.0, 0.0, 0.2, limit=1.0)


# 风干扰估计器
def estimate_wind(vx, vy):
    wind_x = 0.5 * vx if abs(vx) > 0.1 else 0
    wind_y = 0.3 * vy if abs(vy) > 0.1 else 0
    return wind_x, wind_y

# 模拟IMU加速度读数（实际应用中应来自传感器）
def simulate_acceleration(cmd_acc):
    noise = np.random.normal(0, 0.05)  # 模拟噪声
    return cmd_acc + noise


# 主控制函数：三层 PID 控制结构
def controller(state, target, dt):
    # 解包状态
    pos_x, pos_y, pos_z, roll, pitch, yaw = state
    target_x, target_y, target_z, target_yaw = target
    vx = vy = vz = 0.0  # 当前速度（实际中应传入）

    # 坐标误差
    ex_world = target_x - pos_x
    ey_world = target_y - pos_y
    ez = target_z - pos_z

    # 世界坐标 → 机体坐标
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    ex = cos_yaw * ex_world + sin_yaw * ey_world
    ey = -sin_yaw * ex_world + cos_yaw * ey_world

    # 风干扰补偿
    wind_x, wind_y = estimate_wind(vx, vy)

    # ---------- 第一层：位置 PID → 期望速度 ----------
    vx_des = pid_pos_x.compute(ex, dt)
    vy_des = pid_pos_y.compute(ey, dt)
    vz_des = pid_pos_z.compute(ez, dt)

    # ---------- 第二层：速度 PID → 期望加速度 ----------
    error_vx = vx_des - vx - wind_x
    error_vy = vy_des - vy - wind_y
    error_vz = vz_des - vz

    ax_des = pid_vel_x.compute(error_vx, dt)
    ay_des = pid_vel_y.compute(error_vy, dt)
    az_des = pid_vel_z.compute(error_vz, dt)

    # ---------- 第三层：加速度 PID → 推力指令 ----------
    # 模拟当前加速度（真实系统应从IMU获取）
    ax_actual = simulate_acceleration(0)
    ay_actual = simulate_acceleration(0)
    az_actual = simulate_acceleration(0)

    thrust_x = pid_thrust_x.compute(ax_des - ax_actual, dt)
    thrust_y = pid_thrust_y.compute(ay_des - ay_actual, dt)
    thrust_z = pid_thrust_z.compute(az_des - az_actual, dt)

    # Yaw 控制
    error_yaw = target_yaw - yaw
    error_yaw = np.arctan2(np.sin(error_yaw), np.cos(error_yaw))
    yaw_rate_cmd = pid_yaw.compute(error_yaw, dt)

    return (thrust_x, thrust_y, thrust_z, yaw_rate_cmd)



