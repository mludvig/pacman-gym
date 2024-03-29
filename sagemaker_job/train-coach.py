from sagemaker_rl.coach_launcher import SageMakerCoachPresetLauncher

class Launcher(SageMakerCoachPresetLauncher):

    def default_preset_name(self):
        """This points to a .py file that configures everything about the RL job.
        It can be overridden at runtime by specifying the RLCOACH_PRESET hyperparameter.
        """
        return 'PacMan_A3C'

    def map_hyperparameter(self, name, value):
        """Here we configure some shortcut names for hyperparameters that we expect to use frequently.
        Essentially anything in the preset file can be overridden through a hyperparameter with a name
        like "rl.agent_params.algorithm.etc".
        """
        # maps from alias (key) to fully qualified coach parameter (value)
        mapping = {
                      "discount": "rl.agent_params.algorithm.discount",
                      "evaluation_episodes": "rl.evaluation_steps:EnvironmentEpisodes",
                      "improve_steps": "rl.improve_steps:TrainingSteps",
                      "learning_rate": "rl.agent_params.network_wrappers['main'].learning_rate",
                  }
        if name in mapping:
            self.apply_hyperparameter(mapping[name], value)
        else:
            super().map_hyperparameter(name, value)

if __name__ == '__main__':
    Launcher.train_main()
