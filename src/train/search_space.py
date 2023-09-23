from hyperopt import hp

param_tuning_dict = {
    "llama": {
        "learning_rate": hp.loguniform(1e-4, 1e-2),
        "warm_step": hp.randint(50, 105),
    },
}
