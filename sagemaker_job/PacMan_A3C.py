from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.agents.policy_optimization_agent import PolicyGradientRescaler
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.architectures.layers import Dense, Conv2d
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.base_parameters import EmbedderScheme, MiddlewareScheme

from pacman_env import env_params, preset_validation_params

from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.base_parameters import PresetValidationParameters
from rl_coach.environments.gym_environment import ObservationSpaceType

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(100)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(16)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.GAE
agent_params.algorithm.discount = 0.99
agent_params.algorithm.apply_gradients_every_x_episodes = 10
agent_params.algorithm.num_steps_between_gradient_updates = 10
agent_params.algorithm.gae_lambda = 1
agent_params.algorithm.beta_entropy = 0.01

agent_params.network_wrappers['main'].optimizer_type = 'Adam'
agent_params.network_wrappers['main'].learning_rate = 0.0001

agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Conv2d(32, 2, 1), Conv2d(32, 2, 2), Dense(64)]
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].activation_function = 'relu'
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].input_rescaling = {'image': 1.0, 'vector': 1.0, 'tensor': 1.0}
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Empty

########
# Test #
########
preset_validation_params.num_workers = 8

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
