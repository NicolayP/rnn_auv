from tqdm import tqdm
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
from bagpy import bagreader
import shutil
from scipy.spatial.transform import Rotation as R

from manifpy import SE3, SE3Tangent

import argparse

renameLabelsS = {'pose.pose.position.x': "x",
                 'pose.pose.position.y': "y",
                 'pose.pose.position.z': "z",
                 'pose.pose.orientation.x': "qx",
                 'pose.pose.orientation.y': "qy",
                 'pose.pose.orientation.z': "qz",
                 'pose.pose.orientation.w': "qw",
                 'twist.twist.linear.x': "Iu",
                 'twist.twist.linear.y': "Iv",
                 'twist.twist.linear.z': "Iw",
                 'twist.twist.angular.x': "Ip",
                 'twist.twist.angular.y': "Iq",
                 'twist.twist.angular.z': "Ir"}

renameLabelsA = {'wrench.force.x': "Fx",
                 'wrench.force.y': "Fy",
                 'wrench.force.z': "Fz",
                 'wrench.torque.x': "Tx",
                 'wrench.torque.y': "Ty",
                 'wrench.torque.z': "Tz"}

def clean_bag(dataDir, outDir, n=500, freq=0.1):
    '''
    Main function. Cleans a set of bags contained in directory dataDir.
    The bag is cleaned, resized and resampled. If the bag is corrupted
    it is moved to a directory, within the dataDir, named corrupted.

    inputs:
    -------
        - dataDir string, the directory containing the bags to clean.
        - outDir string, saving directory for the csv files.
        - n int, the number of samples to keep. The size 
            of the bag is min(n, len(bag))
        - freq float, the time frequency of the resampled 
            bag. Expressed in seconds.

    '''
    corruptDir = join(dataDir, 'corrupted')
    if not exists(corruptDir):
        mkdir(corruptDir)
    if not exists(outDir):
        os.makedirs(outDir)
    files = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
    t = tqdm(files, desc="Cleaning", ncols=150, colour="green", postfix={"corrupted:": None})
    for f in t:
        bagFile = join(dataDir, f)
        name = os.path.splitext(f)[0]
        bagDir = join(dataDir, name)
        corrupt = join(dataDir, "corrupted", f)
        if exists(bagFile):
            traj = traj_from_bag(bagFile, renameLabelsS, renameLabelsA, freq)
            try:
                
                if (n is not None) and n < len(traj):
                    traj = traj[:n]
                columns = traj.columns
            except:
                t.set_postfix({"corrupted:": f"{f}"})
                os.rename(bagFile, corrupt)
                if exists(bagDir):
                    shutil.rmtree(bagDir)
                continue
        pd.DataFrame(data=traj, columns=columns).to_csv(os.path.join(outDir, name + ".csv"))

def traj_from_bag(path, rds, rda, freq):
    '''
        Extracts a path from a rosbag and returns a
        pandas dataframe resampled at frequency freq.

        inputs:
        -------
            - path str, the path to the rosbag.
            - rds dict, dict whose entry is the old
                state name and value is the new state name 
            - rda dict, dict whose entry is the old
                action name and value is the new action name
            - freq float, the frequency expressed in seconds
        
        outputs:
        -------
            - newly renamed and resampled dataframe containing
                state and action.
    '''
    bag = bagreader(path, verbose=False)
    dfs = pd.read_csv(bag.message_by_topic("/rexrov2/pose_gt"))
    dfa = pd.read_csv(bag.message_by_topic("/thruster_input"))
    traj = df_traj(dfs, rds, dfa, rda, freq)
    traj = traj.set_index(np.arange(len(traj)))
    return traj

def df_traj(dfs, rds, dfa, rda, freq):
    '''
        Resamples action and state dataframe, renames the entries
        and add euler and rotation matrix rotational representation.
        the euler angle are in radians.

        inputs:
        -------
            - dfs, pd.dataframe of the state evolution.
            - rds, dict whose entry is the old
                state name and value is the new state name.
                The old name matches entries in dfs.
            - dfa, pd.dataframe of the action evolution.
            - rds, dict whose entry is the old
                aciton name and value is the new acton name.
                The old name matches entries in dfa.
            - freq float, the desired frequency of the data
                expressed in seconds.

        outputs:
        --------
    '''
    trajS = resample(dfs, rds, freq)
    trajA = resample(dfa, rda, freq)
    quats = trajS.loc[:, ['qx', 'qy', 'qz', 'qw']].to_numpy()
    r = R.from_quat(quats)
    euler = r.as_euler('xyz', False)
    mat = r.as_matrix()
    rot_vec = r.as_rotvec()

    trajS['roll'] = euler[:, 0]
    trajS['pitch'] = euler[:, 1]
    trajS['yaw'] = euler[:, 2]

    trajS['r00'] = mat[:, 0, 0]
    trajS['r01'] = mat[:, 0, 1]
    trajS['r02'] = mat[:, 0, 2]

    trajS['r10'] = mat[:, 1, 0]
    trajS['r11'] = mat[:, 1, 1]
    trajS['r12'] = mat[:, 1, 2]

    trajS['r20'] = mat[:, 2, 0]
    trajS['r21'] = mat[:, 2, 1]
    trajS['r22'] = mat[:, 2, 2]

    trajS['rv0'] = rot_vec[:, 0]
    trajS['rv1'] = rot_vec[:, 1]
    trajS['rv2'] = rot_vec[:, 2]

    b_vel = get_body_vel(trajS)
    trajS['Bu'] = b_vel[:, 0]
    trajS['Bv'] = b_vel[:, 1]
    trajS['Bw'] = b_vel[:, 2]
    trajS['Bp'] = b_vel[:, 3]
    trajS['Bq'] = b_vel[:, 4]
    trajS['Br'] = b_vel[:, 5]

    #dv = compute_dv(trajS)

    #trajS['du'] = dv[:, 0]
    #trajS['dv'] = dv[:, 1]
    #trajS['dw'] = dv[:, 2]
    
    #trajS['dp'] = dv[:, 3]
    #trajS['dq'] = dv[:, 4]
    #trajS['dr'] = dv[:, 5]

    traj = pd.concat([trajS, trajA], axis=1)
    return traj

def resample(df, rd, freq):
    '''
        Resamples and renames a dataframe with the 
        right entries name at the desired frequency.

        inputs:
        -------
            - df, pd.dataframe: dataframe containing the data 
                to resample.
            - rd, dict: dict whose key are entries in the dataframe
                and values are the new desired entires' name.
            - freq, float: the desired frequency of the data
                expressed in seconds.
    '''
    labels = list(rd.keys())
    labels.append('Time')
    df = df.loc[:, labels]
    # relative time of a traj as all the trajs are captured in the same gazebo instance.
    df['Time'] = df['Time'] - df['Time'][0]
    df['Time'] = pd.to_datetime(df['Time'], unit='s').round('ms')
    traj = df.copy()
    traj.index = df['Time']
    traj.rename(columns=rd, inplace=True)
    traj.drop('Time', axis=1, inplace=True)
    traj = traj.resample('ms').interpolate('linear').resample(f'{freq}S').interpolate()
    return traj

def get_body_vel(traj):
    pose = traj.loc[:, ['x', 'y', 'z', 'rv0', 'rv1', 'rv2']].to_numpy()
    I_vel = traj.loc[:, ['Iu', 'Iv', 'Iw', 'Ip', 'Iq', 'Ir']].to_numpy()
    B_vel = np.zeros(shape=I_vel.shape)
    for i, (p, Iv) in enumerate(zip(pose, I_vel)):
        X = SE3.Identity() + SE3Tangent(p)
        Xinv = X.inverse()
        adj = Xinv.adj()
        B_vel[i] = (adj @ Iv[..., None])[..., 0]
    return B_vel

def compute_dv(df):
    '''
        Compute $\DeltaV$ in lie-algebra aka at the origin.

        inputs:
        -------
            - df, pd.dataframe: A dataframe containing the trajectory.
    '''
    vel = df.loc[:, ['u', 'v', 'w', 'p', 'q', 'r']].to_numpy()
    adj = adjoint(df)
    vel_I = np.matmul(adj, vel[..., None])[..., 0]
    dv = np.zeros(shape=vel.shape)
    # dv_{t+1} = vel_inertial_{t+1} - vel_inertial_{t}
    dv[1:] = vel_I[1:] - vel[:-1]
    return dv

def adjoint(df):
    r = df.loc[:, ['r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22']].to_numpy().reshape((-1, 3, 3))
    
    rho = df.loc[:, ['x', 'y', 'z']].to_numpy()
    rot_vec = df.loc[:, ['rv0', 'rv1', 'rv2']].to_numpy()


    print("head: ", df.head(2))
    exit()

    print("rho:       ", rho.shape)
    print("rot_vec:   ", rot_vec.shape)

    v = v_se2(rot_vec)

    print("v(theta):  ", v.shape)

    # expand for matmul, reduce to get shape [k, 3]
    t = np.matmul(v, rho[..., None])[..., 0]

    print("t:         ", t.shape)


    skewT = np.zeros((r.shape[0], 3, 3))

    skewT[:, 0, 1] = - t[:, 2]
    skewT[:, 1, 0] = t[:, 2]

    skewT[:, 0, 2] = t[:, 1]
    skewT[:, 2, 0] = - t[:, 1]

    skewT[:, 1, 2] = - t[:, 0]
    skewT[:, 2, 1] = t[:, 0]

    tmp = np.matmul(skewT, r)
    adj = np.zeros((r.shape[0], 6, 6))
    adj[:, 0:3, 0:3] = r
    adj[:, 3:6, 3:6] = r
    adj[:, 0:3, 3:6] = tmp
    return adj

def v_se2(rot_vec):
    k = rot_vec.shape[0]
    theta_norm = np.linalg.norm(rot_vec, axis=-1)

    skew_theta = np.zeros((k, 3, 3))
    skew_theta[:, 0, 1] = -rot_vec[:, 2]
    skew_theta[:, 1, 0] = rot_vec[:, 2]
    skew_theta[:, 0, 2] = rot_vec[:, 2]
    skew_theta[:, 2, 0] = -rot_vec[:, 2]
    skew_theta[:, 1, 2] = -rot_vec[:, 2]
    skew_theta[:, 2, 1] = rot_vec[:, 2]

    skew_theta_square = np.matmul(skew_theta, skew_theta)

    I = np.eye(3)

    s = np.sin(theta_norm)
    c = np.cos(theta_norm)

    v = I + ((1-c)/np.power(theta_norm, 2))[..., None, None] * skew_theta + ((theta_norm - s)/np.power(theta_norm, 3))[..., None, None] * skew_theta_square

    return v

def parse_arg():
    parser = argparse.ArgumentParser(prog="clean_bags",
                                     description="Cleans and resamples a set of rosbags\
                                        and saves the into a csv file.")

    parser.add_argument('-d', '--datadir', type=str, default=None,
                        help="dir containing the bags to clean.")

    parser.add_argument('-o', '--outdir', type=str, default=".",
                        help="output directory for cleaned up bags.")

    parser.add_argument("-f", "--frequency", type=float,
                        help="Desired transition frequency in the\
                              output bag(s). Default 0.1s. The frequency is expressed in seconds",
                        default=0.1)

    parser.add_argument('-s', '--steps', type=int,
                        help='number of steps to keep in the bag', default=500)

    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    if args.datadir is not None:
        clean_bag(args.datadir, args.outdir, args.steps, args.frequency)
        return
    print("No datadir provided, nothing to clean")

if __name__ == "__main__":
    main()