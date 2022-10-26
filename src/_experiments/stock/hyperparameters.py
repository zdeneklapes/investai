A2C_PARAMS = (False, "a2c", None, 50_000)

DDPG_PARAMS = (False, "ddpg", None, 50_000)
PPO_PARAMS = (
    False,
    "ppo",
    {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    },
    30_000,
)

TD3_PARAMS = (False, "sac", {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}, 30_000)

SAC_HYPERPARAMS = (
    True,
    "sac",
    {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
    30_000,
)
