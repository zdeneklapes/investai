import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

initial_amount = 100000
board_lot = 20
iteration = 3

data_1 = pd.read_csv(
    "E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv"
)
# dji_ = data_market.Adj_Close
# x = np.arange(0, 4500, 20)
# plt.xticks(x, fontsize=14, )
# plt.yticks(fontsize=14, )
# plt.grid(linestyle="--")
# plt.xlim(4000, 4500)
# plt.ylim(18000, 28000)
# plt.plot(dji_,'r')
# plt.show()
# 1761  3268
scar = 2
normalized_market_diff = np.ones([4459, 1]) * scar
normalized_market_diff[45:60] = 1 / scar
normalized_market_diff[165:180] = 1 / scar
normalized_market_diff[330:440] = 1 / scar
normalized_market_diff[500:545] = 1 / scar
normalized_market_diff[1690:2050] = 1 / scar
normalized_market_diff[2590:2705] = 1 / scar
# normalized_market_diff[3650:3705] = 1/scar
# normalized_market_diff[3760:3800] = 1/scar
# normalized_market_diff[3940:3985] = 1/scar
# normalized_market_diff[4290:4400] = 1/scar
normalized_market_diff[3268 + 300 : 3268 + 520] = 1 / scar
normalized_market_diff[3268 + 980 : 3268 + 1080] = 1 / scar

equal_4711_list = list(data_1.tic.value_counts() == 4711)
names = data_1.tic.value_counts().index

# select_stocks_list = ['NKE','KO']
select_stocks_list = list(names[equal_4711_list]) + ["NKE", "KO"]

data_2 = data_1[data_1.tic.isin(select_stocks_list)][
    ~data_1.datadate.isin(["20010912", "20010913"])
]

data_3 = data_2[["iid", "datadate", "tic", "prccd", "ajexdi"]]

data_3["adjcp"] = data_3["prccd"] / data_3["ajexdi"]

train_data = data_3[(data_3.datadate > 20010000) & (data_3.datadate < 20140000)]

train_daily_data = []

for date in np.unique(train_data.datadate):
    train_daily_data.append(train_data[train_data.datadate == date])

dji = pd.read_csv(
    "E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/^DJI.csv"
)
train_dji = dji[dji["Date"] > "2001-01-01"]
dji_price = train_dji["Adj Close"]
dji_date = train_dji["Date"]
daily_return = dji_price.pct_change(1)
daily_return = daily_return[1:]
daily_return.reset_index()
total_amount = initial_amount
account_growth = list()
account_growth.append(initial_amount)
for i in range(len(daily_return)):
    total_amount = total_amount * daily_return.iloc[i] + total_amount
    account_growth.append(total_amount)


class StockEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, day=0, money=10, scope=1):
        self.day = day

        # buy or sell maximum 5 shares
        self.action_space = spaces.Box(
            low=-board_lot, high=board_lot, shape=(28,), dtype=np.int8
        )

        # [money]+[prices 1-28]+[owned shares 1-28]+market_alpha
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(57,))

        # # [money]+[prices 1-28]+[owned shares 1-28]
        # self.observation_space = spaces.Box(low=0, high=np.inf, shape = (5,))

        self.data = train_daily_data[self.day]

        self.alpha = normalized_market_diff[self.day]

        self.terminal = False

        self.state = (
            [initial_amount] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        )
        self.reward = 0

        self.asset_memory = [initial_amount]

        self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        if self.state[index + 29] > 0:
            self.state[0] += self.state[index + 1] * min(
                abs(action), self.state[index + 29]
            )
            self.state[index + 29] -= min(abs(action), self.state[index + 29])
        else:
            pass

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))
        self.state[0] -= self.state[index + 1] * min(available_amount, action)
        # print(min(available_amount, action))

        self.state[index + 29] += min(available_amount, action)

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= 3268
        # 1741 1509
        # print(actions)
        if normalized_market_diff[self.day] == 1 / scar:
            actions = -2 * np.abs(actions)

        if self.terminal:
            x = np.arange(0, 4460, 400)
            plt.xticks(
                x,
                fontsize=14,
            )
            plt.yticks(
                fontsize=14,
            )
            plt.grid(linestyle="--")
            plt.xlim(0, 4460)

            plt.plot(self.asset_memory, "r")
            plt.plot(account_growth, "k")

            plt.savefig(
                "E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/DQN/iter_{} amou_{} lot_{}.png".format(
                    iteration, initial_amount, board_lot
                )
            )
            plt.close()
            print(
                "total_reward:{}".format(
                    self.state[0]
                    + sum(np.array(self.state[1:29]) * np.array(self.state[29:57]))
                    - initial_amount
                )
            )
            # print(len(self.state));

            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:57]))))
            return (
                self.state,
                self.reward / initial_amount,
                self.terminal,
                {},
                self.alpha,
            )

        else:
            # print(np.array(self.state[1:29]))

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:29]) * np.array(self.state[29:57])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print(actions.size)
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = train_daily_data[self.day]

            # print("stock_shares:{}".format(self.state[29:57]))
            self.state = (
                [self.state[0]]
                + self.data.adjcp.values.tolist()
                + list(self.state[29:57])
            )
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:29]) * np.array(self.state[29:57])
            )
            # print("end_total_asset:{}".format(end_total_asset))

            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))

            self.asset_memory.append(end_total_asset)

        return self.state, self.reward / initial_amount, self.terminal, {}, self.alpha

    def reset(self):
        self.asset_memory = [initial_amount]
        self.day = 0
        self.data = train_daily_data[self.day]
        self.state = (
            [initial_amount] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        )

        # iteration += 1
        return self.state

    def render(self, mode="human"):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
