import datetime
import pandas as pd
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from DRFW_TQC_env import PandaPickEnv

class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.epoch_reset_count = 0
        self.epoch_success_count = 0
        self.epoch_success_count_last = 0
        self.epoch_reward_count = 0
        self.epoch_reward_count_last = 0
        self.epoch_reset_count_interval = 0
        self.epoch_reward_count_interval = 0
        self.epoch_success_count_interval = 0
        self.epoch_reset_count_last = 0

        self.epoch_collsion_count = 0
        self.epoch_collsion_count_last = 0
        self.epoch_collsion_count_interval = 0

        self.epoch_outtime_count = 0

        self.epoch_first_success_count = 0

        self.epoch_angle_error_count = 0
        self.epoch_angle_error_count_last = 0
        self.epoch_angle_errorcount_interval = 0


        self.log_dir = f'../{datetime_now}_success'
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'epoch_stats.csv')
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=["epoch", "success_count", "reset_count", "success_rate","reward","collsion","outtime","first_success","angle_error"])
            df.to_csv(self.log_file, index=False)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "epoch_reset" in info:
                current_reset = info["epoch_reset"] - 2
                self.epoch_reset_count = current_reset
                self.epoch_reset_count_interval += current_reset - self.epoch_reset_count_last
                self.epoch_reset_count_last = current_reset

            if "epoch_success" in info:
                current_success = info["epoch_success"]
                self.epoch_success_count = current_success
                self.epoch_success_count_interval += current_success - self.epoch_success_count_last
                self.epoch_success_count_last = current_success
            if "epoch_reward" in info:
                current_reward = info["epoch_reward"]
                self.epoch_reward_count = current_reward
                self.epoch_reward_count_interval += current_reward - self.epoch_reward_count_last
                self.epoch_reward_count_last = current_reward

            if "epoch_collsion" in info:
                current_collsion = info["epoch_collsion"]
                self.epoch_collsion_count = current_collsion

            if "epoch_outtime" in info:
                current_outttime = info["epoch_outtime"]
                self.epoch_outtime_count = current_outttime

            if "epoch_first_success" in info:
                current_first_success = info["epoch_first_success"]
                self.epoch_first_success_count = current_first_success

            if "epoch_angle_error" in info:
                current_angle_error = info["epoch_angle_error"]
                self.epoch_angle_error_count = current_angle_error

        if self.num_timesteps % (200000 // 101) == 0:
            self._save_epoch_stats()
        return True

    def _save_epoch_stats(self):
        success_rate = self.epoch_success_count_interval / self.epoch_reset_count_interval if self.epoch_reset_count_interval > 0 else 0

        df = pd.DataFrame([{
            "epoch": self.num_timesteps,
            "success_count": self.epoch_success_count_interval,
            "reset_count": self.epoch_reset_count_interval,
            "success_rate": success_rate,
            "reward": self.epoch_reward_count_interval,
            "collsion":self.epoch_collsion_count,
            "outtime":self.epoch_outtime_count,
            "first_success":self.epoch_first_success_count,
            "angle_error":self.epoch_angle_error_count
        }])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        self.epoch_reset_count_interval = 0
        self.epoch_success_count_interval = 0
        self.epoch_reward_count_interval = 0

def make_env(env_idx,datetime_now):
    def convert():
        env_class = PandaPickEnv(render=False)
        log_dir = f"../vec_log/{datetime_now}_env/{env_idx}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        env = Monitor(env_class, filename=log_dir, info_keywords=('step', 'reward'))
        return env
    return convert



if __name__ == "__main__":
    num_envs = 1
    datetime_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    env = DummyVecEnv([make_env(i, datetime_now) for i in range(num_envs)])
    from DRFW_TQC_policy import CustomTQCPolicy
    from DRFW_TQC_DL2 import DL2_TQC
    model = DL2_TQC(
        policy=CustomTQCPolicy,
        env=env,
        verbose=1,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256, 256,256]),
    )
    print(model.policy)
    try:
        eval_callback = CustomEvalCallback(env,
                                     best_model_save_path=f'../model_save/{datetime_now}',
                                     eval_freq=500,
                                     n_eval_episodes=2,
                                     deterministic=True,
                                     render=False
                                     )
        log_file = f'../vec_log/{datetime_now}_env/datetime_now_success_log.csv'
        if not os.path.exists(log_file):
            df = pd.DataFrame(columns=["epoch", "success_count", "reset_count", "success_rate"])
            df.to_csv(log_file, index=False)
        total_timesteps = 220000//300
        for epoch in range(300):
            total_success_count = 0
            total_reset_count = 0
            model.learn(total_timesteps=total_timesteps,
                        callback=eval_callback,
                        reset_num_timesteps = False)
            print(f"datetime_now={datetime_now}")
    except KeyboardInterrupt:
        print(f"datetime_now={datetime_now}")
