# -*- coding: utf-8 -*-
import wandb


def wandb_summary(info: dict) -> None:
    for k, v in info.items():
        wandb.run.summary[k] = v
