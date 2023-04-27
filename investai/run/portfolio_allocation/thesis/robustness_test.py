# -*- coding: utf-8 -*-

from shared.program import Program


def find_the_model():
    # Download the trained models from wandb parameters for the best model

    # get parameters for the best models, (must be train on the same algorithm and same data)

    pass


def call_training():
    # get data from find_the_model

    # Compare if all models have the same (similar) test/total_reward (trained
    # on the same data, with same hyperparameters)

    pass


def test_model(model):
    total_rewards = {}
    for i in range(1000):
        # Model test
        # Store total reward
        pass


def test_robustness():
    trained_models = call_training()

    # Multiple times call testing period for each model
    for model in trained_models:
        test_model(model)


def main():
    program = Program()


if __name__ == '__main__':
    main()
