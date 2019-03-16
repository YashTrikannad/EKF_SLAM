from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2


def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    ###
    # Implement the vehicle model and its Jacobian you derived.
    ###

    # vt = u[0]
    # at = u[1]
    # a = ekf_state['x'][0]
    # b = ekf_state['x'][1]
    # th = ekf_state['x'][2]
    #
    # tmp1 = a * np.sin(th) + b * np.cos(th)
    # tmp2 = a * np.cos(th) - b * np.sin(th)
    #
    # G = np.array([[0, 0, -dt*vt*(np.sin(th) + 1/vehicle_params['L']*np.tan(at)*tmp2)],
    #              [0, 0, dt*vt*(np.cos(th) - 1/vehicle_params['L']*np.tan(at)*tmp1)],
    #               [0, 0, 0]])
    #
    # motion = [0, 0, 0]
    #
    # motion[0] = dt * vt * (np.cos(th) - 1 / vehicle_params['L'] * np.tan(at) * tmp1)
    # motion[1] = dt * vt * (np.sin(th) + 1 / vehicle_params['L'] * np.tan(at) * tmp2)
    # motion[2] = dt * vt / vehicle_params['L'] * np.tan(at)

    xv = ekf_state['x'][0]
    yv = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    ve = u[0]
    alpha = u[1]
    H = vehicle_params['H']
    L = vehicle_params['L']
    a = vehicle_params['a']
    b = vehicle_params['b']

    vc = ve/(1 - (np.tan(alpha) * H /L))

    G = np.array([[1, 0, -dt * (vc * np.sin(phi) + (vc * np.tan(alpha) * (a * np.cos(phi) - b * np.sin(phi))) / L)],
                  [0, 1, dt * (vc * np.cos(phi) - (vc * np.tan(alpha) * (b * np.cos(phi) + a * np.sin(phi))) / L)],
                  [0, 0, 1]])

    motion = np.array([[dt * (vc * np.cos(phi) - vc * np.tan(alpha) * (a * np.sin(phi) + b * np.cos(phi)) / L),
                        dt * (vc * np.sin(phi) - vc * np.tan(alpha) * (a * np.cos(phi) - b * np.sin(phi)) / L),
                        dt * vc * np.tan(alpha) / L]])

    return motion, G


def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    ###
    # Implement the propagation
    ###

    motion, G = motion_model(u, dt, ekf_state, vehicle_params)
    x_ = ekf_state['x'][0:3]
    x = x_ + motion
    ekf_state['x'][:3] = x
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    dim = ekf_state['x'].shape[0] - 3

    # Noise
    R_t = np.zeros((3, 3))
    R_t[0, 0], R_t[1, 1], R_t[2, 2] = sigmas['xy']*sigmas['xy'], sigmas['xy']*sigmas['xy'], sigmas['phi']*sigmas['phi']

    Gt = np.eye(dim+3)
    Gt[:3, :3] = G

    Rx = np.eye(dim+3)
    Rx[:3, :3] = R_t
    ekf_state['P'] = np.matmul(np.matmul(Gt, ekf_state['P']), np.transpose(Gt)) + Rx
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    
    ###
    # Implement the GPS update.
    ###

    # Qt = sigmas['gps']*sigmas['gps']*np.eye(2)
    #
    # Sinv = slam_utils.invert_2x2_matrix(ekf_state['P'][:2, :2] + Qt)
    # Kt = np.matmul(ekf_state['P'][:2, :2], Sinv)
    #
    # r = np.transpose(gps - ekf_state['x'][:2])
    #
    # distance = np.matmul(np.matmul(np.transpose(r), Sinv), r)
    #
    # if chi2.ppf(0.999, 2) > distance:
    #
    #     kt_mul_innovation = np.matmul(Kt, r)
    #     ekf_state['x'][:2] = ekf_state['x'][:2] + kt_mul_innovation
    #     ekf_state['P'][:2, :2] = np.matmul((np.eye(2) - kt_mul_innovation), ekf_state['P'][:2, :2])
    #     ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    #

    P = ekf_state['P']
    P_mat = np.matrix(P)
    r = np.transpose([gps - ekf_state['x'][:2]])
    H_mat = np.matrix(np.zeros([2, P.shape[0]]))
    H_mat[0, 0] = 1
    H_mat[1, 1] = 1
    R_mat = np.matrix(np.zeros([2, 2]))
    R_mat[0, 0] = sigmas['gps'] ** 2
    R_mat[1, 1] = sigmas['gps'] ** 2
    S_mat = H_mat * P_mat * H_mat.T + R_mat
    d = (np.matrix(r)).T * np.matrix(slam_utils.invert_2x2_matrix(np.array(S_mat))) * np.matrix(r)
    if d <= chi2.ppf(0.999, 2):
        K_mat = P_mat * H_mat.T * np.matrix(slam_utils.invert_2x2_matrix(np.array(S_mat)))
        ekf_state['x'] = ekf_state['x'] + np.squeeze(np.array(K_mat * np.matrix(r)))
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        ekf_state['P'] = slam_utils.make_symmetric(
            np.array((np.matrix(np.eye(P.shape[0])) - K_mat * H_mat) * P_mat))

    return ekf_state


def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    
    ###
    # Implement the measurement model and its Jacobian you derived
    ###

    delta_x = ekf_state['x'][2 * landmark_id + 3] - ekf_state['x'][0]
    delta_y = ekf_state['x'][2 * landmark_id + 4] - ekf_state['x'][1]
    delta = [delta_x, delta_y]
    q = np.matmul(np.transpose(delta), delta)
    zhat = np.zeros((2, 1))
    sqrt_q = np.sqrt(q)
    zhat[0] = sqrt_q
    zhat[1] = np.arctan2(delta_y, delta_x) - ekf_state['x'][2]

    H_low = np.array([[-sqrt_q*delta_x, -sqrt_q*delta_y, 0, sqrt_q*delta_x, sqrt_q*delta_y],
                     [delta_y, -delta_x, -q, -delta_y, delta_x]])
    # dim = len(ekf_state['x'])
    # Fx = np.zeros((5, dim))
    #
    # H = np.matmul(H_low, Fx)

    H = H_low
    return zhat, H


def initialize_landmark(ekf_state, tree):
    '''turnin -c ese650 -p project2 slam.py
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    ###
    # Implement this function.
    ###
    current_x = ekf_state['x'][0]
    current_y = ekf_state['x'][1]
    current_th = ekf_state['x'][2]

    measurements_r = tree[0]
    measurements_th = tree[1]

    ekf_state['x'] = np.hstack((ekf_state['x'], current_x + measurements_r*np.cos(measurements_th + current_th),
                                current_y + measurements_r*np.sin(measurements_th + current_th)))

    xh_addtoP = np.zeros((1, 2 * ekf_state['num_landmarks'] + 3))
    yh_addtoP = np.zeros((1, 2 * ekf_state['num_landmarks'] + 3))
    ekf_state['num_landmarks'] = ekf_state['num_landmarks'] + 1
    xv_addtoP = np.zeros((2 * ekf_state['num_landmarks'] + 3, 1))
    yv_addtoP = np.zeros((2 * ekf_state['num_landmarks'] + 3, 1))
    xv_addtoP[2 * ekf_state['num_landmarks'] + 1] = .1
    yv_addtoP[2 * ekf_state['num_landmarks'] + 2] = .1

    ekf_state['P'] = np.hstack((np.vstack((ekf_state['P'], xh_addtoP, yh_addtoP)), xv_addtoP, yv_addtoP))

    return ekf_state


def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        # assoc = np.zeros((len(measurements),1))
        # assoc = np.asarray([-1])
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    n_landmarks = np.int((len(ekf_state['x'])-3)/2)

    for k in range(n_landmarks):



        for j in range(len(measurements[0])):

            q = np.matmul(np.transpose(delta), delta)
            zhat = np.zeros((2, 1))
            sqrt_q = np.sqrt(q)
            zhat[0] = sqrt_q
            zhat[1] = np.arctan2(delta_y, delta_x) - ekf_state['x'][2]

            H_low = np.array([[-sqrt_q * delta_x, -sqrt_q * delta_y, 0, sqrt_q * delta_x, sqrt_q * delta_y],
                              [delta_y, -delta_x, -q, -delta_y, delta_x]])


    return assoc


def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###

    # laser_measurement_model(ekf_state, landmark_id=1)

    # Qt = [[sigmas['range']**2, 0], [0, sigmas['bearing']**2]]
    #
    # for i in range(len(trees)):
    #
    #     j = assoc[i]
    #     if j == -1:
    #         ekf_state = initialize_landmark(ekf_state, trees[i])
    #         j = np.int(len(ekf_state['x'])/2) - 2
    #
    #     zhat, Hlow = laser_measurement_model(ekf_state, j)
    #
    #     P = np.zeros((5, 5))
    #     P[:3, :3] = ekf_state['P'][:3, :3]
    #     P[:3, 3:5] = ekf_state['P'][:3,  2*j: 2*j + 2]
    #     P[3:5, :3] = ekf_state['P'][2 * j: 2 * j + 2, :3]
    #     P[3:5, 3:5] = ekf_state['P'][2*j: 2*j + 2, 2*j: 2*j + 2]
    #
    #     # Kalman Gain Initialization
    #     temp1 = np.matmul(P, np.transpose(Hlow))
    #     temp2 = slam_utils.invert_2x2_matrix(np.matmul(Hlow, temp1))
    #     Kt = np.matmul(temp1, temp2)
    #
    #     # Lidar Landmark j measurement
    #     z = np.zeros((2, 1))
    #     z[0] = trees[j][0]
    #     z[1] = trees[j][1]
    #
    #     # Mean Update
    #     change_ut = np.matmul(Kt, z-zhat)
    #     ekf_state['x'][:3] = ekf_state['x'][:3] + np.squeeze(change_ut[:3])
    #     ekf_state['x'][3 + 2*j: 5 + 2*j] = ekf_state['x'][3 + 2*j: 5 + 2*j] + np.squeeze(change_ut[3:5])
    #
    #     # Covariance Update
    #     change_Pt = np.matmul((np.eye(5) - np.matmul(Kt, Hlow)), P)
    #     ekf_state['P'][:3, :3] = change_Pt[:3, :3]
    #     ekf_state['P'][:3, 2 * j: 2 * j + 2] = change_Pt[:3, 3:5]
    #     ekf_state['P'][2 * j: 2 * j + 2, :3] = change_Pt[3:5, :3]
    #     ekf_state['P'][2 * j: 2 * j + 2, 2 * j: 2 * j + 2] = change_Pt[3:5, 3:5]

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key=lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.4,
        "bearing": 3*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)


if __name__ == '__main__':
    main()
