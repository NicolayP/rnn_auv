import numpy as np
import matplotlib.pyplot as plt

from manifpy import SE3, SE3Tangent
from scipy.spatial.transform import Rotation as R

from RNN_AUV import plot_traj

from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

# use of manifpy
X = SE3(position=np.array([0, 0, 0]), quaternion=np.array([0, 0, 0, 1]))
print(X.log().coeffs())
X.rotation()
R.from_matrix(X.rotation())

traj_gt = []
traj = []

b_initialised = False
dt = 0.05

# create reader instance
with Reader('./test_data_clean/bags/run0.bag') as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/rexrov2/pose_gt':
            msg = deserialize_cdr(ros1_to_cdr(
                rawdata, connection.msgtype), connection.msgtype)

            # print(msg.header.frame_id)

            pose_gt = np.array([msg.pose.pose.position.x,
                                msg.pose.pose.position.y,
                                msg.pose.pose.position.z,
                                msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
            traj_gt.append(pose_gt)

            if not b_initialised:
                pose = SE3(position=pose_gt[0:3], quaternion=pose_gt[3:])
                b_initialised = True
            else:
                # velocity in inertial frame
                dv = np.array([msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y,
                            msg.twist.twist.linear.z,
                            msg.twist.twist.angular.x,
                            msg.twist.twist.angular.y,
                            msg.twist.twist.angular.z])

                # velocity in body frame
                dv = pose.inverse().adj() @ dv
                # dv[0:3] = np.array([msg.twist.twist.linear.x,
                #                     msg.twist.twist.linear.y,
                #                     msg.twist.twist.linear.z])

                # T_w_k = T_w_k-1 * T_k-1_k
                pose = pose + SE3Tangent(dv * dt)

            traj.append(np.concatenate([pose.translation(), pose.quat()]))

            print("pose_gt=", pose_gt)
            print("pose=", pose)


    # # messages() accepts connection filters
    # connections = [x for x in reader.connections if x.topic == '/rexrov2/pose_gt']
    # for connection, timestamp, rawdata in reader.messages(connections=connections):
    #     msg = deserialize_cdr(ros1_to_cdr(
    #         rawdata, connection.msgtype), connection.msgtype)
    #     print(msg.header.frame_id)

traj_gt_np = np.array(traj_gt)
traj_np = np.array(traj)

s_cols = {'x': 0, 'y': 1, 'z': 2, '$theta 1$': 3, '$theta 2$': 4, '$theta 3$': 5}
plot_traj({"Traj": traj_np[:, 0:7], "gt": traj_gt_np[:, 0:7]}, s_cols, traj_np.shape[0], True)
plt.show()

