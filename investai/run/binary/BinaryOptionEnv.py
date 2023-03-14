# -*- coding: utf-8 -*-
import datetime
import math
import random

import numpy as np
from gymnasium import spaces, Env
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from meta.env_fx_trading.util.log_render import render_to_file
from meta.env_fx_trading.util.plot_chart import TradingChart


class BinartOptionEnv(Env):
    metadata = {"render.modes": ["graph", "human", "file", "none"]}

    def __init__(self, df):
        super().__init__()
        self.balance_initial = self.cf.env_parameters("balance")
        self.over_night_cash_penalty = self.cf.env_parameters("over_night_cash_penalty")
        self.asset_col = self.cf.env_parameters("asset_col")
        self.time_col = self.cf.env_parameters("time_col")
        self.random_start = self.cf.env_parameters("random_start")
        self.log_filename = (
            self.cf.env_parameters("log_filename")
            + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            + ".csv"
        )

        self.df = df
        self.df["_time"] = df[self.time_col]
        self.df["_day"] = df["weekday"]
        self.assets = df[self.asset_col].unique()
        self.dt_datetime = df[self.time_col].sort_values().unique()
        self.df = self.df.set_index(self.time_col)
        self.visualization = False

        # --- reset value ---
        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.current_step = 0
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ""
        self.log_header = True
        # --- end reset ---
        self.cached_data = [
            self.get_observation_vector(_dt) for _dt in self.dt_datetime
        ]
        self.cached_time_serial = (
            (self.df[["_time", "_day"]].sort_values("_time")).drop_duplicates()
        ).values.tolist()

        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=0, high=3, shape=(len(self.assets),))
        # first two 3 = balance,current_holding, max_draw_down_pct
        _space = 3 + len(self.assets) + len(self.assets) * len(self.observation_list)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(_space,))
        print(
            f"initial done:\n"
            f"observation_list:{self.observation_list}\n "
            f"assets:{self.assets}\n "
            f"time serial: {min(self.dt_datetime)} -> {max(self.dt_datetime)} length: {len(self.dt_datetime)}"
        )
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _history_df(self, i):
        pass

    def _take_action(self, actions, done):
        # action = math.floor(x),
        # profit_taken = math.ceil((x- math.floor(x)) * profit_taken_max - stop_loss_max )
        # _actions = np.floor(actions).astype(int)
        # _profit_takens =
        # np.ceil((actions - np.floor(actions)) *self.cf.symbol(self.assets[i],"profit_taken_max")).astype(int)
        _action = 2
        _profit_taken = 0
        rewards = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        # need use multiply assets
        for i, x in enumerate(actions):
            self._o = self.get_observation(self.current_step, i, "Open")
            self._h = self.get_observation(self.current_step, i, "High")
            self._l = self.get_observation(self.current_step, i, "Low")
            self._c = self.get_observation(self.current_step, i, "Close")
            self._t = self.get_observation(self.current_step, i, "_time")
            self._day = self.get_observation(self.current_step, i, "_day")
            _action = math.floor(x)
            rewards[i] = self._calculate_reward(i, done)
            if self.cf.ticker(self.assets[i], "limit_order"):
                self._limit_order_process(i, _action, done)
            if (
                _action in (0, 1)
                and not done
                and self.current_holding[i]
                < self.cf.ticker(self.assets[i], "max_current_holding")
            ):
                # generating PT based on action fraction
                _profit_taken = math.ceil(
                    (x - _action) * self.cf.ticker(self.assets[i], "profit_taken_max")
                ) + self.cf.ticker(self.assets[i], "stop_loss_max")
                self.ticket_id += 1
                if self.cf.ticker(self.assets[i], "limit_order"):
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._l if _action == 0 else self._h,
                        "SL": self.cf.ticker(self.assets[i], "stop_loss_max"),
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.ticker(self.assets[i], "transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": -1,
                        "CloseStep": -1,
                    }
                    self.transaction_limit_order.append(transaction)
                else:
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._c,
                        "SL": self.cf.ticker(self.assets[i], "stop_loss_max"),
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.ticker(self.assets[i], "transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": self.current_step,
                        "CloseStep": -1,
                    }
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= self.cf.ticker(self.assets[i], "transaction_fee")
                    self.transaction_live.append(transaction)

        return sum(rewards)

    def _calculate_reward(self, i, done):
        _total_reward = 0
        _max_draw_down = 0
        for tr in self.transaction_live:
            if tr["Symbol"] == self.assets[i]:
                _point = self.cf.ticker(self.assets[i], "point")
                # cash discount overnight
                if self._day > tr["DateDuration"]:
                    tr["DateDuration"] = self._day
                    tr["Reward"] -= self.cf.ticker(self.assets[i], "over_night_penalty")

                if tr["Type"] == 0:  # buy
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] - tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] + tr["PT"] / _point
                    if done:
                        p = (self._c - tr["ActionPrice"]) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._l <= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._h >= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:  # still open
                        self.current_draw_downs[i] = int(
                            (self._l - tr["ActionPrice"]) * _point
                        )
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                            self.current_draw_downs[i] < 0
                            and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                elif tr["Type"] == 1:  # sell
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] + tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] - tr["PT"] / _point
                    if done:
                        p = (tr["ActionPrice"] - self._c) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._h >= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._l <= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:
                        self.current_draw_downs[i] = int(
                            (tr["ActionPrice"] - self._h) * _point
                        )
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                            self.current_draw_downs[i] < 0
                            and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                if _max_draw_down > self.max_draw_downs[i]:
                    self.max_draw_downs[i] = _max_draw_down

        return _total_reward

    def _limit_order_process(self, i, _action, done):
        for tr in self.transaction_limit_order:
            if tr["Symbol"] == self.assets[i]:
                if tr["Type"] != _action or done:
                    self.transaction_limit_order.remove(tr)
                    tr["Status"] = 3
                    tr["CloseStep"] = self.current_step
                    self.transaction_history.append(tr)
                elif (tr["ActionPrice"] >= self._l and _action == 0) or (
                    tr["ActionPrice"] <= self._h and _action == 1
                ):
                    tr["ActionStep"] = self.current_step
                    self.current_holding[i] += 1
                    self.balance -= self.cf.ticker(self.assets[i], "transaction_fee")
                    self.transaction_limit_order.remove(tr)
                    self.transaction_live.append(tr)
                    self.tranaction_open_this_step.append(tr)
                elif (
                    tr["LimitStep"]
                    + self.cf.ticker(self.assets[i], "limit_order_expiration")
                    > self.current_step
                ):
                    tr["CloseStep"] = self.current_step
                    tr["Status"] = 4
                    self.transaction_limit_order.remove(tr)
                    self.transaction_history.append(tr)

    def _manage_tranaction(self, tr, _p, close_price, status=1):
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)
        tr["Status"] = status
        tr["CloseTime"] = self._t
        self.balance += int(tr["Reward"])
        self.total_equity -= int(abs(tr["Reward"]))
        self.tranaction_close_this_step.append(tr)
        self.transaction_history.append(tr)

    def step(self, actions):
        # Execute one time step within the environment
        self.current_step += 1
        done = self.balance <= 0 or self.current_step == len(self.dt_datetime) - 1
        if done:
            self.done_information += f"Episode: {self.episode} Balance: {self.balance} Step: {self.current_step}\n"
            self.visualization = True
        reward = self._take_action(actions, done)
        if self._day > self.current_day:
            self.current_day = self._day
            self.balance -= self.over_night_cash_penalty
        if self.balance != 0:
            self.max_draw_down_pct = abs(sum(self.max_draw_downs) / self.balance * 100)

            # no action anymore
        obs = (
            [self.balance, self.max_draw_down_pct]
            + self.current_holding
            + self.current_draw_downs
            + self.get_observation(self.current_step)
        )
        return (
            np.array(obs).astype(np.float32),
            reward,
            done,
            {"Close": self.tranaction_close_this_step},
        )

    def get_observation(self, _step, _iter=0, col=None):
        if col is None:
            return self.cached_data[_step]
        if col == "_day":
            return self.cached_time_serial[_step][1]

        elif col == "_time":
            return self.cached_time_serial[_step][0]
        col_pos = -1
        for i, _symbol in enumerate(self.observation_list):
            if _symbol == col:
                col_pos = i
                break
        assert col_pos >= 0
        return self.cached_data[_step][_iter * len(self.observation_list) + col_pos]

    def get_observation_vector(self, _dt, cols=None):
        cols = self.observation_list
        v = []
        for a in self.assets:
            subset = self.df.query(
                f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"'
            )
            assert not subset.empty
            v += subset.loc[_dt, cols].tolist()
        assert len(v) == len(self.assets) * len(cols)
        return v

    def reset(self):
        # Reset the state of the environment to an initial state
        self.seed()

        if self.random_start:
            self.current_step = random.choice(range(int(len(self.dt_datetime) * 0.5)))
        else:
            self.current_step = 0

        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ""
        self.log_header = True
        self.visualization = False

        _space = (
            [self.balance, self.max_draw_down_pct]
            + [0] * len(self.assets)
            + [0] * len(self.assets)
            + self.get_observation(self.current_step)
        )
        return np.array(_space).astype(np.float32)

    def render(self, mode="human", title=None, **kwargs):
        # Render the environment to the screen
        if mode in ("human", "file"):
            printout = mode == "human"
            pm = {
                "log_header": self.log_header,
                "log_filename": self.log_filename,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.balance_initial,
                "tranaction_close_this_step": self.tranaction_close_this_step,
                "done_information": self.done_information,
            }
            render_to_file(**pm)
            if self.log_header:
                self.log_header = False
        elif mode == "graph" and self.visualization:
            print("plotting...")
            p = TradingChart(self.df, self.transaction_history)
            p.plot()

    def close(self):
        pass

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
