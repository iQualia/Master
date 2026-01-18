import contextlib
import os

import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2


@contextlib.contextmanager
def _suppress_output():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _load_urdf_silent(*args, **kwargs):
    with _suppress_output():
        body_id = p.loadURDF(*args, **kwargs)
    if body_id < 0:
        raise RuntimeError(f"Failed to load URDF: {args[0]}")
    return body_id


class CubeTrackingEnv(gym.Env):
    def __init__(self, mode='headless'):
        super(CubeTrackingEnv, self).__init__()

        # Connect to PyBullet with GUI or headless mode
        if mode == 'gui':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Show GUI controls
        else:  # headless
            self.physics_client = p.connect(p.DIRECT)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load environment
        self.plane_id = _load_urdf_silent("plane.urdf")
        self.robot_id = _load_urdf_silent(
            "/home/Master/mycobot_urdf_copy.urdf",
            [-0.1, 0, 0],
            useFixedBase=True
        )
        self.cube_id = _load_urdf_silent("/home/Master/Cube.urdf", [0.3, 0, 0.2])
        object_id = _load_urdf_silent(
            "/home/Master/Inspection_Station.urdf",
            useFixedBase=True
        )
        rpy_rotation = [0, 0, 3.142]  # Roll, Pitch, Yaw in radians
        quaternion = p.getQuaternionFromEuler(rpy_rotation)
        position = [0.3, 0, 0.0]  # Set desired position (X, Y, Z)
        p.resetBasePositionAndOrientation(object_id, position, quaternion)
        position_collision = [0.3, 0, 0.03]

        #Tower_Horizontal

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Tower_Horizontal.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )

        #Tower_Vertical
        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Tower_Vertical.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )


        #Wall

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Wall.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )

        #Platform

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="Platform.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )

        #PiBOX
        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="PiBox.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )
        #Bottom

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Bottom.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )
        #camerabox
        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/camerabox.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )

        # Camera parameters
        self.camera_width = 480
        self.camera_height = 640
        self.fov = 60
        self.aspect_ratio = self.camera_width / self.camera_height
        self.near_plane = 0.01
        self.far_plane = 10

        # End effector link index
        self.end_effector_index = 6

        # Define observation and action spaces
        # Note: Robot may have more joints (e.g., gripper), but we only control the first 7 arm joints
        self.num_joints = 7  # MyCobot arm has 7 controllable joints (gripper joint is fixed, excluded)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + 3 + self.num_joints + 3,),  # ee_pos(3) + cube_pos(3) + joints(7) + distance(1) + alignment(1) + visible_flag(1) = 16
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.max_steps = 300  # or 500
        self.current_step = 0


        # Reset environment
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.plane_id = _load_urdf_silent("plane.urdf")
        self.robot_id = _load_urdf_silent(
            "/home/Master/mycobot_urdf_copy.urdf",
            [-0.1, 0, 0],
            useFixedBase=True
        )

        # Initialize robot to ready pose - camera tilted toward cube spawn area
        # Only joint 4 is set to -0.5 rad, all others at 0 (URDF default)
        ready_pose = [0, 0, 0, 0, -0.5, 0, 0]
        for i, angle in enumerate(ready_pose):
            p.resetJointState(self.robot_id, i, angle)

        self.cube_id = _load_urdf_silent("/home/Master/Cube.urdf", [0.3, 0, 0.2])
        object_id = _load_urdf_silent(
            "/home/Master/Inspection_Station.urdf",
            useFixedBase=True
        )
        rpy_rotation = [0, 0, 3.142]  # Roll, Pitch, Yaw in radians
        quaternion = p.getQuaternionFromEuler(rpy_rotation)
        position = [0.3, 0, 0.0]  # Set desired position (X, Y, Z)
        p.resetBasePositionAndOrientation(object_id, position, quaternion)
        position_collision = [0.3, 0, 0.03]

                #Tower_Horizontal

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Tower_Horizontal.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )

        #Tower_Vertical
        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Tower_Vertical.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )


        #Wall

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Wall.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )

        #Platform

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="Platform.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )

        #PiBOX
        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="PiBox.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )
        #Bottom

        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/Bottom.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )
        #camerabox
        shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/Master/camerabox.stl",
            meshScale=[1, 1, 1]  # Scale if needed
        )
        body_id = p.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,  # Visual shape (if already loaded)
            basePosition=position_collision,  # Position
            baseOrientation=p.getQuaternionFromEuler(rpy_rotation)  # Orientation
        )



        # Randomize cube position
        cube_position = np.random.uniform(low=[0.25, -0.06, 0.1], high=[0.32, 0.06, 0.1])
        p.resetBasePositionAndOrientation(self.cube_id, cube_position, [0, 0, 0, 1])
        self.current_step = 0


        return self._get_observation()

    def _get_observation(self):
        # End effector pose (always observable)
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        ee_pos = np.array(ee_state[0])
        ee_ori = np.array(ee_state[1])

        # Joint positions (always observable)
        joint_positions = np.array([p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)])

        # Check if cube is visible in camera view
        camera_data = self._get_camera_view()
        segmentation_image = camera_data[4]
        cube_visible = np.any(segmentation_image == self.cube_id)

        if cube_visible:
            # Cube IS visible: provide real measurements
            cube_pos, cube_ori = p.getBasePositionAndOrientation(self.cube_id)
            cube_pos = np.array(cube_pos)
            cube_ori = np.array(cube_ori)

            # Distance between end effector and cube
            distance = np.linalg.norm(ee_pos - cube_pos)

            # Alignment between camera Z-axis and cube X-axis
            ee_rot = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
            cube_rot = np.array(p.getMatrixFromQuaternion(cube_ori)).reshape(3, 3)
            camera_z_axis = ee_rot[:, 2]
            cube_x_axis = cube_rot[:, 0]

            camera_z_axis /= np.linalg.norm(camera_z_axis)
            cube_x_axis /= np.linalg.norm(cube_x_axis)
            alignment = np.dot(camera_z_axis, cube_x_axis)

            visible_flag = 1.0  # Cube is visible
        else:
            # Cube NOT visible: mask measurements with zeros
            cube_pos = np.zeros(3)      # [0, 0, 0] - no position information
            distance = 0.0              # No distance measurement
            alignment = 0.0             # No alignment measurement
            visible_flag = 0.0          # Flag: not visible

        # Final observation vector with visibility flag
        return np.concatenate([ee_pos, cube_pos, joint_positions, [distance, alignment, visible_flag]])

    def step(self, action):
        # Apply actions to joints (velocity control)
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=action[i],
                force=50
            )

        # Step simulation multiple times for stability
        # At 1/240s timestep, 4 substeps = ~60Hz control frequency
        for _ in range(4):
            p.stepSimulation()

        # Get observation
        obs = self._get_observation()

        # End effector (camera) state
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
        end_effector_pos = np.array(end_effector_state[0])
        end_effector_orientation = np.array(end_effector_state[1])

        # Cube state
        cube_state = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_state[0])
        cube_orientation = np.array(cube_state[1])

        # Compute rotation matrices
        end_effector_rotation_matrix = np.array(p.getMatrixFromQuaternion(end_effector_orientation)).reshape(3, 3)
        cube_rotation_matrix = np.array(p.getMatrixFromQuaternion(cube_orientation)).reshape(3, 3)

        # Camera Z-axis and Cube X-axis
        camera_z_axis = end_effector_rotation_matrix[:, 2]  # Z-axis of the camera
        cube_x_axis = cube_rotation_matrix[:, 0]  # X-axis of the cube

        # Normalize vectors
        camera_z_axis = camera_z_axis / np.linalg.norm(camera_z_axis)
        cube_x_axis = cube_x_axis / np.linalg.norm(cube_x_axis)

        # Dot product for alignment
        alignment = np.dot(camera_z_axis, cube_x_axis)

        # Compute distance
        distance = np.linalg.norm(end_effector_pos - cube_pos)

        # Reward function
        reward = -np.square(distance) # Penalize distance
        #reward += alignment * 10  # Reward for alignment
        # info["reward_config"] = "C1"

        # Check if cube is visible (extract from observation's visibility flag)
        # visible_flag is the last element of observation (index -1)
        cube_visible = (obs[-1] > 0.5)  # visible_flag = 1.0 if visible, 0.0 if not
        if cube_visible:
            reward += 10  # Reward if cube is visible
        else:
            reward -= 10  # Penalize if cube is not visible

        # Penalize collisions
        #contact_points = p.getContactPoints(bodyA=self.robot_id)
#         contact_points = [
#         c for c in p.getContactPoints(bodyA=self.robot_id)
#         if c[3] != c[4]  # linkIndexA != linkIndexB
# ]
#         if contact_points:
#             reward -= 10  # Penalize collisions

        # Termination condition
        self.current_step += 1
        done = distance < 0.05 or reward < -40 or self.current_step >= self.max_steps


        # Compute success based on achievable criteria (adjusted for robot limitations)
        # Oracle achieves ~0.08-0.15m distance at final step, so 0.15m is a reasonable threshold
        success = (distance < 0.15) and cube_visible and (alignment > 0.5)

        # Debug info
        info = {
            "distance_to_cube": distance,
            "alignment": alignment,
            "cube_visible": cube_visible,
            "success": success,  # Required for curriculum advancement and evaluation
            # "collision_detected": len(contact_points) > 0,
            "reward_config" : "C1"
        }

        # Display camera image for debugging
        # rgb_image = camera_data[2]
        # rgb_image = np.reshape(rgb_image, (self.camera_height, self.camera_width, 4))[:, :, :3]  # Remove alpha channel
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        # cv2.imshow("Camera View", rgb_image)
        # cv2.waitKey(1)

        return obs, reward, done, info


    def _get_camera_view(self):
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
        end_effector_pos = np.array(end_effector_state[0])
        end_effector_orientation = np.array(end_effector_state[1])

        rotation_matrix = np.array(p.getMatrixFromQuaternion(end_effector_orientation)).reshape(3, 3)

        # Camera offset (tuned from gripper_tuner_realtime.py)
        cam_offset_local = np.array([0.0, -0.018, 0.0])
        cam_offset_world = rotation_matrix @ cam_offset_local
        camera_pos = end_effector_pos + cam_offset_world

        forward_vector = rotation_matrix[:, 2]  # Z-axis as forward vector
        up_vector = rotation_matrix[:, 1]
        up_vector = -up_vector# Y-axis as up vector
        target_position = camera_pos + forward_vector

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect_ratio,
            nearVal=self.near_plane,
            farVal=self.far_plane
        )

        return p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()

import time
from stable_baselines3.common.callbacks import BaseCallback

# class TrainingProgressCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(TrainingProgressCallback, self).__init__(verbose)
#         self.start_time = None  # Initialize the start time

#     def _on_training_start(self) -> None:
#         # Record the start time when training begins
#         self.start_time = time.time()

#     def _on_step(self) -> bool:
#         # Calculate elapsed time
#         elapsed_time = time.time() - self.start_time

#         # Log progress every 1000 steps
#         if self.n_calls % 10 == 0:  # Every 1000 steps
#             print(f"Step: {self.n_calls}, Time Elapsed: {elapsed_time:.2f} seconds")

#             # Log episode rewards and length
#             ep_rewards = self.locals.get("rewards")
#             if ep_rewards:
#                 avg_reward = sum(ep_rewards) / len(ep_rewards)
#                 print(f"Average Reward: {avg_reward:.2f}")
#         return True
class TrainingProgressCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=10):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.start_time = None
        self.log_interval = log_interval  # Number of steps between logs
        self.rewards = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Save reward for averaging
        reward = self.locals.get("rewards")
        if reward:
            self.rewards.extend(reward)

        # Access info dictionary from environment
        infos = self.locals.get("infos")
        if infos and isinstance(infos, list):
            # Just get the first environment's info (assuming single env)
            info = infos[0]

            # Log every `log_interval` steps
            if self.n_calls % self.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                avg_reward = sum(self.rewards[-self.log_interval:]) / max(len(self.rewards[-self.log_interval:]), 1)

                print(f"Step: {self.n_calls}, Time Elapsed: {elapsed_time:.2f} sec")
                print(f"  ➤ Avg Reward (last {self.log_interval} steps): {avg_reward:.2f}")
                print(f"  ➤ Distance to Cube: {info.get('distance_to_cube'):.4f}")
                print(f"  ➤ Alignment: {info.get('alignment'):.4f}")
                print(f"  ➤ Cube Visible: {info.get('cube_visible')}")
                print(f"  ➤ Collision Detected: {info.get('collision_detected')}")

        return True


if __name__ == "__main__":
    # env = CubeTrackingEnv()
    # obs = env.reset()

    # for _ in range(5000):
    #     action = env.action_space.sample()  # Random action
    #     obs, reward, done, info = env.step(action)
    #     print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
    #     if done:
    #         obs = env.reset()

    # env.close()

    from stable_baselines3 import SAC

    # Create environment
    env = CubeTrackingEnv()

    # Create the model
    model = SAC("MlpPolicy", env, verbose=1, ent_coef="auto",tensorboard_log="./sac_tensorboard/",device="cuda" )

    # Instantiate the callback
    callback = TrainingProgressCallback(verbose=1)

    # Train with the callback
    model.learn(total_timesteps=100000, callback=callback)

    # Save the model
    model.save("sac_cube_tracking")
    
