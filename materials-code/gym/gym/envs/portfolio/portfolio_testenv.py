import math

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
from gym import spaces
from gym.utils import seeding

initial_amount = 100000
board_lot = 20
iteration = 3

dji = pd.read_csv(
    "E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/^DJI.csv"
)
test_dji = dji[dji["Date"] > "2014-01-01"]
dji_price = test_dji["Adj Close"]
dji_date = test_dji["Date"]
daily_return = dji_price.pct_change(1)
daily_return = daily_return[1:]
daily_return.reset_index()


total_amount = initial_amount
account_growth = list()
account_growth.append(initial_amount)
for i in range(len(daily_return) - 10):
    total_amount = total_amount * daily_return.iloc[i] + total_amount
    account_growth.append(total_amount)
# np.savetxt("account_growth.txt", account_growth)
np.savetxt("DJIA.txt", account_growth)

# Minimum variance
data = pd.read_csv(
    "E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/daily_return.csv"
)
name_list = [
    "Date",
    "WMT",
    "XOM",
    "MCD",
    "UTX",
    "DWDP",
    "MMM",
    "WBA",
    "NKE",
    "CAT",
    "PFE",
    "INTC",
    "MSFT",
    "AXP",
    "GS",
    "PG",
    "V",
    "DIS",
    "KO",
    "AAPL",
    "CSCO",
    "TRV",
    "IBM",
    "JNJ",
    "MRK",
    "BA",
    "UNH",
    "HD",
    "VZ",
    "CVX",
    "JPM",
]
data.columns = name_list
test = data[data["Date"] > 20140101]
test = test[test["Date"] < 20181001]
stock_return = test.iloc[:, 1:]
cov_mat = stock_return.cov()
cov_mat_annual = cov_mat * 252
number = 500
n = 30
random_p = np.empty((number, n + 2))
np.random.seed(123)
ticker_list = [
    "WMT",
    "XOM",
    "MCD",
    "UTX",
    "DWDP",
    "MMM",
    "WBA",
    "NKE",
    "CAT",
    "PFE",
    "INTC",
    "MSFT",
    "AXP",
    "GS",
    "PG",
    "V",
    "DIS",
    "KO",
    "AAPL",
    "CSCO",
    "TRV",
    "IBM",
    "JNJ",
    "MRK",
    "BA",
    "UNH",
    "HD",
    "VZ",
    "CVX",
    "JPM",
]
for i in range(number):
    random9 = np.random.random(n)
    random_weight = random9 / np.sum(random9)

    mean_return = stock_return.mul(random_weight, axis=1).sum(axis=1).mean()
    annual_return = (1 + mean_return) ** 252 - 1

    random_volatility = np.sqrt(
        np.dot(random_weight.T, np.dot(cov_mat_annual, random_weight))
    )

    random_p[i][:30] = random_weight
    random_p[i][30] = annual_return
    random_p[i][31] = random_volatility

RandomPortfolios = pd.DataFrame(random_p)
RandomPortfolios.columns = [ticker + "_weight" for ticker in ticker_list] + [
    "Returns",
    "Volatility",
]
min_index = RandomPortfolios.Volatility.idxmin()
RandomPortfolios.plot("Volatility", "Returns", kind="scatter", alpha=0.3)
x = RandomPortfolios.loc[min_index, "Volatility"]
y = RandomPortfolios.loc[min_index, "Returns"]
numstocks = 30
GMV_weights = np.array(RandomPortfolios.iloc[min_index, 0:numstocks])
StockReturns = pd.DataFrame()
StockReturns["Portfolio_GMV"] = stock_return.mul(GMV_weights, axis=1).sum(axis=1)
daily_return = StockReturns["Portfolio_GMV"]
adr = daily_return.mean()  # average daily return
aar = (adr + 1) ** 252 - 1  # average anuual return
# print('Min average anuual return', aar)
dstd = daily_return.std()  # daily std
astd = dstd * math.sqrt(252)  # annual std
# print('anuanl std minimum variance', astd)
initial_amount = 10000
total_amount = initial_amount
account_growth = list()
account_growth.append(initial_amount)
for i in range(len(daily_return)):
    total_amount = total_amount * daily_return.iloc[i] + total_amount
    account_growth.append(total_amount)
# print('final value minimum variance', account_growth[-6])
np.save("min.npy", account_growth)


# Mean variance
risk_free = 0
RandomPortfolios["Sharpe"] = (
    RandomPortfolios.Returns - risk_free
) / RandomPortfolios.Volatility
max_index = RandomPortfolios.Sharpe.idxmax()
RandomPortfolios.plot("Volatility", "Returns", kind="scatter", alpha=0.3)
x = RandomPortfolios.loc[max_index, "Volatility"]
y = RandomPortfolios.loc[max_index, "Returns"]
MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:numstocks])
StockReturns["Portfolio_MSR"] = stock_return.mul(MSR_weights, axis=1).sum(axis=1)
daily_return = StockReturns["Portfolio_MSR"]
adr = daily_return.mean()  # average daily return
aar = (adr + 1) ** 252 - 1  # average anuual return
# print('Mean average anuual return mean variance', aar)
dstd = daily_return.std()  # daily std
astd = dstd * math.sqrt(252)  # annual std
# print('anuanl std mean variance', astd)
initial_amount = 10000
total_amount = initial_amount
account_growth = list()
account_growth.append(initial_amount)
for i in range(len(daily_return)):
    total_amount = total_amount * daily_return.iloc[i] + total_amount
    account_growth.append(total_amount)
# print('final value mean variance', account_growth[-6])
np.save("mean.npy", account_growth)


data_1 = pd.read_csv(
    "E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv"
)
# data_market = pd.read_csv('E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/zxstock/Data_Daily_Stock_Dow_Jones_30/DJI_market.csv')
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

# train_data = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
test_data = data_3[data_3.datadate > 20140000]

test_daily_data = []

for date in np.unique(test_data.datadate):
    test_daily_data.append(test_data[test_data.datadate == date])

# whole_data = train_daily_data+test_daily_data


class StockTestEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, day=0, money=10, scope=1):
        self.day = day
        # self.money = money

        # buy or sell maximum 5 shares
        self.action_space = spaces.Box(
            low=-board_lot, high=board_lot, shape=(28,), dtype=np.int8
        )

        # # buy or sell maximum 5 shares
        # self.action_space = spaces.Box(low = -5, high = 5,shape = (2,),dtype=np.int8)

        # [money]+[prices 1-28]+[owned shares 1-28]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(57,))

        # # [money]+[prices 1-28]+[owned shares 1-28]
        # self.observation_space = spaces.Box(low=0, high=np.inf, shape = (5,))

        self.data = test_daily_data[self.day]
        self.alpha = normalized_market_diff[self.day + 3267]  # 3267

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
        self.terminal = self.day >= 1189
        # 685
        # self.terminal = self.day >= 1761
        # print(actions)
        if normalized_market_diff[self.day + 3267] == 1 / scar:
            actions = -2 * np.abs(actions)

        if self.terminal:
            plt.plot(self.asset_memory, "r")
            plt.plot(account_growth, "k")
            # np.savetxt("asmemory_amou_{}lot_{}.txt".format(initial_amount,board_lot), self.asset_memory)
            np.savetxt("Adaptive DDPG.txt", self.asset_memory)

            print(self.asset_memory[-1] - account_growth[-1])
            # print(account_growth[-1])
            plt.savefig(
                "E:/Anaconda3/envs/tensorflow/Lib/site-packages/gym/envs/portfolio/Data_Daily_Stock_Dow_Jones_30/DQN/test_{} amou_{} lot_{}.png".format(
                    iteration, initial_amount, board_lot
                )
            )
            plt.close()
            print(
                "total_reward:{}".format(
                    self.state[0]
                    + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))
                    - initial_amount
                )
            )
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
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = test_daily_data[self.day]
            # self.money = self.state[0]

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
        self.data = test_daily_data[self.day]
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
