import random
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import os
from DRFW_TQC_reward import compute_custom_reward
class PandaPickEnv(gym.Env):
    def __init__(self, render=False):
        super(PandaPickEnv, self).__init__()
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 120.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.plane_id = p.loadURDF("../DRFW_TQC_simulation/plane/plane.urdff", useFixedBase=True, flags=flags
                                   )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, False)
        self.ridge = p.loadURDF(
            "../DRFW_TQC_simulation/1_15_first_ridge.urdf",
            [0.7, 0.2, 0.3],
            [0,0,0,1],
            useFixedBase=True,
            flags=flags
        )
        first_leaf_path = "../DRFW_TQC_simulation/leaf/first_leaf.urdf"
        self.first_leaf = p.loadURDF(
            first_leaf_path,
            [0.6, 0.3, 0.3],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        second_leaf_path = "../DRFW_TQC_simulation/leaf/second_leaf.urdf"
        self.second_leaf = p.loadURDF(
            second_leaf_path,
            [0.7, 0.3, 0.25],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        third_leaf_path = "../DRFW_TQC_simulation/leaf/third_leaf.urdf"
        self.third_leaf = p.loadURDF(
            third_leaf_path,
            [0.7, 0.2, 0.25],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        forth_leaf_path = "../DRFW_TQC_simulation/leaf/forth_leaf.urdf"
        self.forth_leaf = p.loadURDF(
            forth_leaf_path,
            [0.6, 0.3, 0.3],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        start_pos_panda = [0, 0, 0.2]
        start_ori_panda = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF("../DRFW_TQC_simulation/fixed/fixed.urdf", [-0.05, 0, 0], useFixedBase=True, flags=flags)
        self.panda_id = p.loadURDF(
            "/home/aaa/.conda/envs/xxx/lib/python3.10/site-packages/pybullet_data/franka_panda/panda.urdf",
            start_pos_panda,
            start_ori_panda,
            useFixedBase=True,
            flags=flags
        )
        self.epoch_reward = 0
        self.distance_last = 0
        self.real_distance_last = 0
        self.success_num = 0
        self.rewards = []
        self.step_num = 0
        self.success_num_all = 0
        self.reward = 0
        self.reset_count = 0
        self.reset_panda=[0.0, 0.0, 0.0, -2.546, 0.0 , 3.0 , 0.75]
        self.controllable_joints = [0, 1, 2, 3, 4, 5, 6]
        for j in self.controllable_joints:
            p.resetJointState(self.panda_id, j, self.reset_panda[j])
        self.fruit_urdf_path = "../DRFW_TQC_simulation/1_09_first_straw.urdf"
        if not os.path.exists(self.fruit_urdf_path):
            raise FileNotFoundError("无法找到草莓 URDF 文件")
        self.plant_id_green = p.loadURDF(
            self.fruit_urdf_path,
            [0.6, 0.4, 0.3],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=flags
        )
        self.fruit_urdf_path2 = "../DRFW_TQC_simulation/1_09_first_straw.urdf"
        if not os.path.exists(self.fruit_urdf_path2):
            raise FileNotFoundError("无法找到草莓 URDF 文件")
        self.plant_id2_green = p.loadURDF(
            self.fruit_urdf_path2,
            [0.6, 0.0, 0.3],
            p.getQuaternionFromEuler([0, 0.3, 0]),
            useFixedBase=True,
            flags=flags
        )
        strawberry_positions  = self.load_strawberries(num_strawberries=4)
        self.pick_order_copy = self.pick_order[:]
        self.plant_id = self.pick_order[0]
        self.order_num = len(self.pick_order)
        self.order_num_copy = self.order_num
        pass
        self.fruit_link_name = "branch"
        self.fruit_link_index = self._find_fruit_link_index()
        self.joint_limits_lower = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
        self.joint_limits_upper = np.array([2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671])
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
        ee_pos1, ee_orn1 = p.getLinkState(self.panda_id, 9)[:2]
        ee_pos2, ee_orn2 = p.getLinkState(self.panda_id, 10)[:2]
        self.ee_pos1_init = ee_pos1
        self.ee_pos2_init = ee_pos2
        self.current_step = 0
        self.current_success = False
        self.epoch_success = 0
        self.epoch_reset = 0
        self.current_reward = {}
        self.collsion_time = 0
        self.out_time = 0
        self.first_success_time = 0
        self.until_success_step_num = 0
        self.angle_error = []
        self.reset()

    def load_strawberries(self,num_strawberries):
        strawberry_urdf_path = "../DRFW_TQC_simulation/1_14_second_straw.urdf"
        if not os.path.exists(strawberry_urdf_path):
            raise FileNotFoundError(f"无法找到草莓 URDF 文件: {strawberry_urdf_path}")
        self.plant_ids = []
        self.base_positions = []
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        for i in range(num_strawberries):
            y_position = random.uniform(-0.25, 0.25)
            position = [0.6, y_position, 0.3]
            orientation = [0, 0, 0]
            plant_id = p.loadURDF(
                strawberry_urdf_path,
                position,
                p.getQuaternionFromEuler(orientation),
                useFixedBase=True,
                flags=flags
            )
            self.plant_ids.append(plant_id)
            self.base_positions.append(position)
        self.pick_order = self.plant_ids.copy()
        self.base_postion = sorted(self.base_positions, key=lambda sub: math.sqrt(sum(x * x for x in sub)))
        return self.base_postion

    def _find_fruit_link_index(self):
        fruit_index = -1
        num_joints = p.getNumJoints(self.plant_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.plant_id, i)
            if info[12].decode('utf-8') == self.fruit_link_name:
                fruit_index = i
                break
        if fruit_index < 0:
            raise ValueError("在URDF文件中未找到fruit链接")
        return fruit_index

    def _get_obs(self):
        joint_positions = []
        for j in self.controllable_joints:
            joint_positions.append(p.getJointState(self.panda_id, j)[0])
        joint_positions = np.array(joint_positions)
        fruit_pos, fruit_orn = p.getLinkState(self.plant_id, self.fruit_link_index)[:2]
        fruit_pos = np.array(fruit_pos, dtype=np.float32)
        obs = np.concatenate([joint_positions, fruit_pos], axis=0)
        return obs

    def reset(self, seed=None, options=None):
        self.epoch_reset += 1
        self.rewards = 0
        self.step_num = 0
        self.reward = 0
        for j in self.controllable_joints:
            p.resetJointState(self.panda_id, j, self.reset_panda[j])
        p.resetJointState(self.panda_id, 9, 0.04)
        p.resetJointState(self.panda_id, 10, 0.04)
        self.pick_order = self.pick_order_copy[:]
        self.plant_id = self.pick_order[0]
        self.order_num = len(self.pick_order)
        for i in range(self.order_num_copy):
            random_offset = [random.uniform(-0.05, 0.05) for _ in range(3)]
            new_position = [self.base_postion[i][j] + random_offset[j] for j in range(3)]
            p.resetBasePositionAndOrientation(
                bodyUniqueId=self.pick_order[i],
                posObj=new_position,
                ornObj=[0.0,0.0,0.0,1.0]
            )
            base_plant_id_green = [0.6, 0.4, 0.3]
            base_plant_id2_green = [0.6, 0.0, 0.3]
            random_offset = [random.uniform(-0.05, 0.05) for _ in range(3)]
            base_plant_id_green_new_position = [base_plant_id_green[j] + random_offset[j] for j in range(3)]
            base_plant_id2_green_new_position = [base_plant_id2_green[j] + random_offset[j] for j in range(3)]
            p.resetBasePositionAndOrientation(
                bodyUniqueId=self.plant_id_green,
                posObj=base_plant_id_green_new_position,
                ornObj=[0.0,0.0,0.0,1.0]
            )
            p.resetBasePositionAndOrientation(
                bodyUniqueId=self.plant_id2_green,
                posObj=base_plant_id2_green_new_position,
                ornObj=[0.0,0.0,0.0,1.0]
            )
        obs = self._get_obs()
        info = {}
        self.current_step = 0
        self.current_reward = 0
        self.reset_count += 1
        return obs,{}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_joints = []
        for j in self.controllable_joints:
            current_joints.append(p.getJointState(self.panda_id, j)[0])
        current_joints = np.array(current_joints)
        target_joints = current_joints + action
        target_joints = np.clip(target_joints, self.joint_limits_lower, self.joint_limits_upper)
        for i, j in enumerate(self.controllable_joints):
            p.setJointMotorControl2(self.panda_id, j, p.POSITION_CONTROL, targetPosition=target_joints[i], force=1000)
        for _ in range(4):
            p.stepSimulation()
        obs = self._get_obs()
        reward, terminated, truncated = compute_custom_reward(self,obs)
        self.current_step += 1
        self.current_reward += self.reward
        self.epoch_reward += reward
        info = {
            'step': self.current_step,
            'reward': self.current_reward,
            'epoch_reset': self.epoch_reset,
            'epoch_success': self.epoch_success,
            'epoch_reward':self.epoch_reward,
            'epoch_collsion':self.collsion_time,
            'epoch_outtime':self.out_time,
            'epoch_first_success':self.first_success_time,
            'epoch_angle_error':sum(self.angle_error)/(len(self.angle_error)+0.001)
        }
        self.until_success_step_num += 1
        return obs, reward, terminated,truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()