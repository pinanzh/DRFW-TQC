
import torch as th
import torch.nn as nn
from sb3_contrib.tqc.policies import TQCPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from typing import Any, Dict, List, Optional, Tuple, Type
from gymnasium import spaces
from DRFW_TQC_GFWN import GFWN

LOG_STD_MAX = 2
LOG_STD_MIN = -20
class CustomActor(BasePolicy):
    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = 7
        self.latent_pi = nn.Sequential(
            nn.Linear(features_dim, 256),
            activation_fn(),
            nn.Linear(256, 256),
            activation_fn(),
            GFWN(feature_dim=256),
            nn.Linear(256, 256),
            activation_fn(),
            nn.Linear(256, 256),
            activation_fn(),
        )
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
        self.mu = nn.Linear(256, action_space.shape[0])
        self.log_std = nn.Linear(256, action_space.shape[0])
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)



from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCritic(BaseModel):
    action_space: spaces.Box
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_quantiles: int = 25,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = 7

        self.share_features_extractor = share_features_extractor
        self.q_networks = []
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.quantiles_total = n_quantiles * n_critics

        for i in range(n_critics):
            qf_net = nn.Sequential(
                nn.Linear(features_dim + action_dim, 256),
                activation_fn(),
                nn.Linear(256, 256),
                activation_fn(),
                nn.Linear(256, 256),
                activation_fn(),
                nn.Linear(256, 256),
                activation_fn(),
                nn.Linear(256, n_quantiles),
            )
            self.add_module(f"qf{i}", qf_net)
            self.q_networks.append(qf_net)

    def forward(self, obs: PyTorchObs, action: th.Tensor) -> th.Tensor:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, action], dim=1)
        quantiles = th.stack(tuple(qf(qvalue_input) for qf in self.q_networks), dim=1)
        return quantiles

class CustomTQCPolicy(TQCPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomCritic(**critic_kwargs).to(self.device)