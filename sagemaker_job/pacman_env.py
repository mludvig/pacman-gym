from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters
from rl_coach.environments.gym_environment import ObservationSpaceType

import gym_pacman

###############
# Environment #
###############

env_params = GymVectorEnvironment(level='PacMan-v1')
#env_params.additional_simulator_parameters = {
#    "board_size" : (10, 10),
#    "max_moves" : 200,
#}
# Set the target success
env_params.target_success_rate = -20

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
#preset_validation_params.min_reward_threshold = 150
#preset_validation_params.max_episodes_to_achieve_reward = 400

#################
# Visualization #
#################

vis_params = VisualizationParameters()
vis_params.dump_gifs = True
#vis_params.render = True
#vis_params.native_rendering = True
