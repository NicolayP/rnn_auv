from tqdm import tqdm
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
from bagpy import bagreader
import shutil
from scipy.spatial.transform import Rotation as R

import argparse

renameLabelsS = {'pose.pose.position.x': "x",
                 'pose.pose.position.y': "y",
                 'pose.pose.position.z': "z",
                 'pose.pose.orientation.x': "qx",
                 'pose.pose.orientation.y': "qy",
                 'pose.pose.orientation.z': "qz",
                 'pose.pose.orientation.w': "qw",
                 'twist.twist.linear.x': "Su",
                 'twist.twist.linear.y': "Sv",
                 'twist.twist.linear.z': "Sw",
                 'twist.twist.angular.x': "Sp",
                 'twist.twist.angular.y': "Sq",
                 'twist.twist.angular.z': "Sr"}

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
            try:
                traj = traj_from_bag(bagFile, renameLabelsS, renameLabelsA, freq)
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

    B_vel = get_body_vel(trajS)
    trajS['Bu'] = B_vel[:, 0]
    trajS['Bv'] = B_vel[:, 1]
    trajS['Bw'] = B_vel[:, 2]
    trajS['Bp'] = B_vel[:, 3]
    trajS['Bq'] = B_vel[:, 4]
    trajS['Br'] = B_vel[:, 5]
    
    I_vel = get_inertial_vel(trajS)
    trajS['Iu'] = I_vel[:, 0]
    trajS['Iv'] = I_vel[:, 1]
    trajS['Iw'] = I_vel[:, 2]
    trajS['Ip'] = I_vel[:, 3]
    trajS['Iq'] = I_vel[:, 4]
    trajS['Ir'] = I_vel[:, 5]

    Idv, Bdv = get_dv(trajS)
    trajS['Idu'] = Idv[:, 0]
    trajS['Idv'] = Idv[:, 1]
    trajS['Idw'] = Idv[:, 2]
    trajS['Idp'] = Idv[:, 3]
    trajS['Idq'] = Idv[:, 4]
    trajS['Idr'] = Idv[:, 5]

    trajS['Bdu'] = Bdv[:, 0]
    trajS['Bdv'] = Bdv[:, 1]
    trajS['Bdw'] = Bdv[:, 2]
    trajS['Bdp'] = Bdv[:, 3]
    trajS['Bdq'] = Bdv[:, 4]
    trajS['Bdr'] = Bdv[:, 5]

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
    rotBtoI = traj.loc[:, ['r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22']].to_numpy().reshape(-1, 3, 3)
    rotItoB = np.transpose(rotBtoI, axes=(0, 2, 1))
    Sim_lin_vel = traj.loc[:, ['Su', 'Sv', 'Sw']].to_numpy()
    Sim_ang_vel = traj.loc[:, ['Sp', 'Sq', 'Sr']].to_numpy()

    B_lin_vel = np.matmul(rotItoB, Sim_lin_vel[..., None])[..., 0]
    B_ang_vel = np.matmul(rotItoB, Sim_ang_vel[..., None])[..., 0]

    B_vel = np.concatenate([B_lin_vel, B_ang_vel], axis=-1)
    return B_vel

def get_inertial_vel(traj):
    Sim_lin_vel = traj.loc[:, ['Su', 'Sv', 'Sw']].to_numpy()
    Sim_ang_vel = traj.loc[:, ['Sp', 'Sq', 'Sr']].to_numpy()
    I_ang_vel = Sim_ang_vel

    t = traj.loc[:, ['x', 'y', 'z']].to_numpy()

    skewT = np.zeros((t.shape[0], 3, 3))

    skewT[:, 0, 1] = - t[:, 2]
    skewT[:, 1, 0] = t[:, 2]

    skewT[:, 0, 2] = t[:, 1]
    skewT[:, 2, 0] = - t[:, 1]

    skewT[:, 1, 2] = - t[:, 0]
    skewT[:, 2, 1] = t[:, 0]

    I_lin_vel = Sim_lin_vel + np.matmul(skewT, Sim_ang_vel[..., None])[..., 0]
    I_vel = np.concatenate([I_lin_vel, I_ang_vel], axis=-1)

    return I_vel

def get_dv(traj):
    I_vel = traj.loc[:, ['Iu', 'Iv', 'Iw', 'Ip', 'Iq', 'Ir']].to_numpy()
    Idv = np.zeros(shape=(I_vel.shape))
    Idv[1:] = I_vel[1:] - I_vel[:-1]


    B_vel = traj.loc[:, ['Bu', 'Bv', 'Bw', 'Bp', 'Bq', 'Br']].to_numpy()
    Bdv = np.zeros(shape=(B_vel.shape))
    Bdv[1:] = B_vel[1:] - B_vel[:-1]

    return Idv, Bdv

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
    if args.datadir is None:
        print("No datadir provided, nothing to clean")
        return
    clean_bag(args.datadir, args.outdir, args.steps, args.frequency)
        


if __name__ == "__main__":
    main()