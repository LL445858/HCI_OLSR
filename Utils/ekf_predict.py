# EKF算法，仅仅用于与UKF做对比，实际项目中未用到

import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, P, x):
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = self.f(self.x)
        F = self.F_jacobian(self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q
        return self.x

    def update(self, z):
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        return self.x


def state_transition(x):
    dt = 1
    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    return np.dot(F, x)


def measurement_function(x):
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
    return np.dot(H, x)


def state_transition_jacobian(x):
    dt = 1
    return np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])


def measurement_jacobian(x):
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])


def kalman(measurements):
    Q = np.eye(6) * 5
    R = np.eye(3)
    P = np.eye(6) * 0.1
    num_steps, num_drones, _ = measurements.shape
    predicted_states = np.zeros((num_steps, num_drones, 6))
    ekf_list = []

    for i in range(num_drones):
        x0 = measurements[0, i, :]
        initial_state = np.array([x0[0], x0[1], x0[2], 0, 0, 0])
        ekf = ExtendedKalmanFilter(state_transition, measurement_function, state_transition_jacobian,
                                   measurement_jacobian, Q, R, P, initial_state)
        ekf_list.append(ekf)
        predicted_states[0, i, :] = initial_state

    for t in range(1, num_steps):
        for i in range(num_drones):
            predicted_states[t, i, :] = ekf_list[i].predict()
            ekf_list[i].update(measurements[t, i, :])

    return predicted_states
