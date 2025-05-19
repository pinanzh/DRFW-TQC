import pybullet as p
import numpy as np

def quaternion_angle_error(q_P, q_T):
    q_P = np.array(q_P) / np.linalg.norm(q_P)
    q_T = np.array(q_T) / np.linalg.norm(q_T)
    dot_product_normal = np.clip(np.abs(np.dot(q_P, q_T)), -1.0, 1.0)
    dot_product_flipped = np.clip(np.abs(np.dot(q_P, -q_T)), -1.0, 1.0)
    theta_error_normal = 2 * np.arccos(dot_product_normal)
    theta_error_flipped = 2 * np.arccos(dot_product_flipped)
    theta_error = min(theta_error_normal, theta_error_flipped)
    return 3.1415926-theta_error



def compute_custom_reward(self, obs):
    terminated = False
    fruit_pos = obs[-3:]
    ee_pos1, ee_orn1 = p.getLinkState(self.panda_id, 9)[:2]
    ee_pos2, ee_orn2 = p.getLinkState(self.panda_id, 10)[:2]
    theta_error = quaternion_angle_error([0.0,0.0,0.0,1.0], ee_orn2)
    dist1 = np.linalg.norm(np.array(ee_pos1) - fruit_pos)
    dist2 = np.linalg.norm(np.array(ee_pos2) - fruit_pos)
    dist = (dist1+dist2) / 2
    dist_reward = - ((dist + 0.3) * 10) ** 2 - 150 * max(fruit_pos[2]-max(ee_pos1[2],ee_pos2[2]),0) - 0.01 * theta_error
    self.reward = dist_reward
    self.step_num += 1
    truncated= False
    self.reward -= 2
    if self.step_num > 500:
        terminated = True
        truncated = True
        self.out_time += 1
        return self.reward, terminated, truncated

    success_plant_1 = False
    orient_error = ee_pos1[2] - ee_pos2[2]
    self.reward -=  (orient_error) ** 2 *20
    for i in range(self.order_num):
        if i == 0:
            pass
        else:
            other_target_collion = p.getContactPoints(bodyA=self.panda_id, bodyB=self.pick_order[i])
            if len(other_target_collion) > 0:
                p.resetBasePositionAndOrientation(
                    bodyUniqueId=self.pick_order[i],
                    posObj=[0, 0, 4],
                    ornObj=[0.0, 0.0, 0.0, 1.0]
                )
                success_plant_1 = True
                self.order_num -= 1
                self.pick_order.pop(i)
                if self.pick_order != []:
                    self.plant_id = self.pick_order[0]
                    next_dis, _ = p.getLinkState(self.pick_order[0], 0)[:2]
                    distance1 = sum((a - b) ** 2 for a, b in zip(list(self.ee_pos1_init), list(next_dis))) ** 0.5
                    distance2 = sum((a - b) ** 2 for a, b in zip(list(self.ee_pos2_init), list(next_dis))) ** 0.5
                    dist = (distance1 + distance2) / 2
                    self.reward -= (dist * 10) ** 2
                for i in range(self.order_num_copy - self.order_num):
                    self.reward += 100 / (i + 1)
                for j in self.controllable_joints:
                    p.resetJointState(self.panda_id, j, self.reset_panda[j])
                p.resetJointState(self.panda_id, 9, 0.04)
                p.resetJointState(self.panda_id, 10, 0.04)
                self.reward -= 25.0
                self.collsion_time += 1
                break
    contact_with_first_green = p.getContactPoints(bodyA=self.panda_id, bodyB=self.plant_id_green)
    if len(contact_with_first_green) > 0:
        self.reward -= 100.0
        self.collsion_time += 1
    contact_with_first_green2 = p.getContactPoints(bodyA=self.panda_id, bodyB=self.plant_id2_green)
    if len(contact_with_first_green2) > 0:
        self.reward -= 100.0
        self.collsion_time += 1
    contact_points = p.getContactPoints(bodyA=self.panda_id)
    contact_with_floor = p.getContactPoints(bodyA=self.panda_id, bodyB=self.plane_id)
    if len(contact_with_floor) > 0:
        self.reward -= 100.0
        self.collsion_time += 1
    contact_with_first_leaf = p.getContactPoints(bodyA=self.panda_id, bodyB=self.first_leaf) or p.getContactPoints(bodyA=self.panda_id, bodyB=self.second_leaf) or p.getContactPoints(bodyA=self.panda_id, bodyB=self.third_leaf) or p.getContactPoints(bodyA=self.panda_id, bodyB=self.forth_leaf)
    if len(contact_with_first_leaf) > 0:
        self.reward -= 5.0
        self.collsion_time += 1
    self_collision = p.getContactPoints(bodyA=self.panda_id, bodyB=self.panda_id)
    if len(self_collision) > 0:
        self.reward -= 5.0
        self.collsion_time += 1
    ridge_collision = p.getContactPoints(bodyA=self.panda_id, bodyB=self.ridge)
    if len(ridge_collision) > 0:
        self.reward -= (100.0)
        self.collsion_time += 1
    fail_non_gripper = False
    fail_dist_gripper = False
    fail_final_gripper = False
    if dist < 0.08:
        p.resetJointState(self.panda_id, 9, 0.0)
        p.resetJointState(self.panda_id, 10, 0.0)
        ee_pos1, _ = p.getLinkState(self.panda_id, 9)[:2]
        ee_pos2, _ = p.getLinkState(self.panda_id, 10)[:2]
        dist1 = np.linalg.norm(np.array(ee_pos1) - fruit_pos)
        dist2 = np.linalg.norm(np.array(ee_pos2) - fruit_pos)
        dist = (dist1 + dist2) / 2
        if dist < 0.08:
            p.resetBasePositionAndOrientation(
                bodyUniqueId=self.plant_id,
                posObj=[0,0,4],
                ornObj=[0.0, 0.0, 0.0, 1.0]
            )
            success_plant_1 = True
            self.order_num -= 1
            self.pick_order.pop(0)
            if self.pick_order != []:
                self.plant_id = self.pick_order[0]
                next_dis ,_ = p.getLinkState(self.pick_order[0], 0)[:2]
                distance1 = sum((a - b) ** 2 for a, b in zip(list(self.ee_pos1_init), list(next_dis))) ** 0.5
                distance2 = sum((a - b) ** 2 for a, b in zip(list(self.ee_pos2_init), list(next_dis))) ** 0.5
                dist = (distance1 + distance2) / 2
                self.reward -=  (dist * 10) ** 2
            for i in range(self.order_num_copy - self.order_num):
                self.reward += 500
            for j in self.controllable_joints:
                p.resetJointState(self.panda_id, j, self.reset_panda[j])
            p.resetJointState(self.panda_id, 9, 0.04)
            p.resetJointState(self.panda_id, 10, 0.04)
        else:
            fail_final_gripper = True
    for cp in contact_points:
        linkA = cp[3]
        linkB = cp[4]
        if linkA in [9,10] and linkB == self.fruit_link_index:
            pass
        if linkB == self.fruit_link_index and linkA not in [8,9,10]:
            fail_non_gripper = True

        if linkB != self.fruit_link_index and contact_with_first_leaf == 0:
            self.reward -= (2.0)
            self.collsion_time += 1
    if fail_final_gripper:
        self.reward -=(2.0/(self.order_num_copy-self.order_num+1)**3)
        terminated = True
        return self.reward, terminated, truncated
    if fail_dist_gripper:
        self.reward -= (3.0/(self.order_num_copy-self.order_num+1)**3)
        terminated = True
        return self.reward, terminated, truncated
    if fail_non_gripper:
        self.reward -= (5.0)
    if success_plant_1:
        self.angle_error.append(theta_error)
        self.success_num += 1
        if self.order_num == 0:
            self.success_num_all +=1
            self.epoch_success += 1
            if self.first_success_time == 0:
                self.first_success_time = self.until_success_step_num
            terminated = True
    if self.until_success_step_num >199000:
        print(f'self.collsion_time={self.collsion_time},self.out_time={self.out_time},self.first_success_time={self.first_success_time},self.until_success_step_num={self.until_success_step_num}')
        print(f'sum(self.angle_error)/len(self.angle_error={sum(self.angle_error)/len(self.angle_error)}')
    return self.reward, terminated, truncated
