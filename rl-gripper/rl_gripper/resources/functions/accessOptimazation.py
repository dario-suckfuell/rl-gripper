import optuna
from optuna.storages import RDBStorage

# Load the study
storage = RDBStorage(url="sqlite:///optuna_study.db")
study = optuna.load_study(study_name="SAC_ResNetOpt_01", storage=storage)

# Access the best trial
best_trial = study.best_trial
print("Best hyperparameters: ", best_trial.params)
print("Best value: ", best_trial.value)

# You can also access all trials
all_trials = study.trials
for trial in all_trials:
    print(trial)
