from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.architectures.layers import Dense, Conv2d

from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.base_parameters import EmbedderScheme, MiddlewareScheme

from pacman_env import env_params, preset_validation_params, vis_params

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(50)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = DQNAgentParameters()

# DQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
agent_params.algorithm.discount = 0.618
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(10)  # was 1

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.001     # was 0.00025
#agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Conv2d(32, 2, 1), Conv2d(32, 2, 2), Dense(64)]
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].activation_function = 'relu'
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].input_rescaling = {'image': 1.0, 'vector': 1.0, 'tensor': 1.0}
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Empty

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, schedule_params.improve_steps.num_steps)     # was 1.0, 0.01, 10000

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
