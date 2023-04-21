# -*- coding: utf-8 -*-
import wandb
from itertools import chain

from tqdm import tqdm
import pandas as pd
import empyrical

from shared.program import Program
from shared.reload import reload_module  # noqa
from extra.math.finance.shared.baseline import Baseline  # noqa
from run.shared.memory import Memory


class Reload(Memory):
    def __init__(self, program: Program = None):
        self.program = program
        self.filepath = self.program.args.history_path.as_posix()

    def download_test_history(self):
        api = wandb.Api()
        runs = [
            api.runs(self.program.args.wandb_entity + "/" + self.program.args.wandb_project, filters={"group": group})
            for group in [
                # "sweep-nasfit-2",
                # "sweep-nasfit-3",
                # "sweep-nasfit-4"
                # "sweep-nasfit-5"
                "sweep-nasfit-6"
            ]
        ]
        log_keys = [
            "test/reward/^DJI",
            "test/reward/^GSPC",
            "test/reward/^IXIC",
            "test/reward/^RUT",
            "test/reward/maximum_sharpe_0_1",
            "test/reward/maximum_quadratic_utility_0_1",
            "test/reward/minimum_variance_0_1",
        ]

        #
        all_samples = 100_000
        iterations = tqdm(enumerate(list(chain(*runs))))
        final_df = pd.DataFrame()
        baseline_done = False

        # all rewards
        for idx, run in iterations:  # type: int,  wandb.apis.public.Run
            if run.state == "running" or run.state == "failed":
                continue

            # Baselines
            if not baseline_done:
                baseline_done = True
                for log_key in tqdm(log_keys):
                    df = run.history(samples=all_samples, keys=[log_key])
                    df.rename(columns={log_key: "reward"}, inplace=True)
                    df['id'] = run.id + "_" + log_key.split("/")[-1]
                    df['group'] = run._attrs['group']
                    df['algo'] = "baseline"
                    final_df = pd.concat([final_df, df])

            # Models
            model_key = "test/reward/model"
            df = run.history(samples=all_samples, keys=[model_key])
            df.rename(columns={model_key: "reward"}, inplace=True)
            df['id'] = run.id
            df['group'] = run._attrs['group']
            df['algo'] = run._attrs['config']['algo']
            final_df = pd.concat([final_df, df])

        #
        assert final_df.groupby(['id']).size().unique().size == 1, "All runs should have the same number of samples"
        final_df.to_csv(self.filepath, index=True)
        if self.program.args.project_verbose > 0:
            self.program.log.info(f"History downloaded and saved to {self.filepath}")
        self.df = final_df
        return self.df

    def initialize_stats(self):
        df: pd.DataFrame = self.load_csv(self.filepath)
        self.returns_pivot_df = df.pivot(columns=['id'], values=['reward'])
        self.returns_pivot_df = pd.concat(
            [pd.DataFrame(data=[[0] * self.returns_pivot_df.columns.__len__()], columns=self.returns_pivot_df.columns),
             self.returns_pivot_df]).reset_index(
            drop=True)
        self.cumprod_returns_df = (self.returns_pivot_df + 1).cumprod()

        # TODO: Change index/_step by date
        self.baseline = Baseline(self.program)
        self.baseline.load_csv(self.program.args.baseline_path.as_posix())

        self.returns_pivot_df.index = pd.to_datetime(self.baseline.df['date'].iloc[-self.returns_pivot_df.shape[0]:],
                                                     format="%Y-%m-%d")
        self.cumprod_returns_df.index = pd.to_datetime(
            self.baseline.df['date'].iloc[-self.cumprod_returns_df.shape[0]:],
            format="%Y-%m-%d"
        )

        self.baseline_columns = self.returns_pivot_df.filter(regex="zy8buea3.*").columns.map(lambda x: x[1]).drop(
            'zy8buea3')

        self.returns_pivot_df.columns = self.returns_pivot_df.columns.droplevel(0)
        self.cumprod_returns_df.columns = self.cumprod_returns_df.columns.droplevel(0)

    def get_summary(self):
        # Cumprod
        cumprod_returns_df_without_baseline = self.cumprod_returns_df.drop(columns=self.baseline_columns)
        idx_min = cumprod_returns_df_without_baseline.iloc[-1].argmin()
        idx_max = cumprod_returns_df_without_baseline.iloc[-1].argmax()
        id_min = cumprod_returns_df_without_baseline.iloc[-1].index[idx_min]
        id_max = cumprod_returns_df_without_baseline.iloc[-1].index[idx_max]

        # Sharpe
        print("Sharpe")
        print(empyrical.sharpe_ratio(self.returns_pivot_df[id_min]))
        print(empyrical.sharpe_ratio(self.returns_pivot_df[id_max]))

        # Beta
        print("Beta")
        print(empyrical.beta(self.returns_pivot_df[id_min], self.returns_pivot_df['zy8buea3_^DJI']))
        print(empyrical.beta(self.returns_pivot_df[id_max], self.returns_pivot_df['zy8buea3_^DJI']))

        # Max drawdown
        print("Max drawdown")
        print(empyrical.max_drawdown(self.returns_pivot_df[id_min]))
        print(empyrical.max_drawdown(self.returns_pivot_df[id_max]))

    def plot_returns(self):
        pass
        # rows = 4
        # plt.subplot(rows, 1, 1)
        # pf.plotting.plot_rolling_returns(returns_pivot_df[id_min], returns_pivot_df[('reward', 'zy8buea3_^DJI')])
        # fig: plt.figure = sns.lineplot(data=df_cumprod)
        # plt.tight_layout()
        # plt.show()
        # plt.subplot(rows, 1, 2)
        # pf.plotting.plot_rolling_returns(returns_pivot_df[id_max], returns_pivot_df[('reward', 'zy8buea3_^DJI')])
        # plt.show()
        # plt.subplot(rows, 1, 3)
        # pf.plotting.plot_returns(returns_pivot_df[id_min])
        # plt.show()
        # plt.subplot(rows, 1, 4)
        # pf.plotting.plot_returns(returns_pivot_df[id_max])
        # plt.show()

    def plot_details(self):
        pass

    # rows = 2
    # plt.subplot(rows, 1, 1)
    # pf.plot_annual_returns(returns_pivot_df[id_min])
    # plt.subplot(rows, 1, 2)
    # pf.plot_monthly_returns_dist(returns_pivot_df[id_min])
    # plt.subplot(rows, 1, 3)
    # pf.plot_monthly_returns_heatmap(returns_pivot_df[id_min])
    # plt.subplot(rows, 1, 4)
    # pf.plot_return_quantiles(returns_pivot_df[id_min])
    # plt.subplot(rows, 1, 5)
    # pf.plot_rolling_beta(returns_pivot_df[id_min], returns_pivot_df[('reward', 'zy8buea3_^DJI')])
    # plt.subplot(rows, 1, 6)
    # pf.plot_rolling_sharpe(returns_pivot_df[id_min])
    # plt.subplot(rows, 1, 7)

    def plot_drawdown(self):
        pass


# pf.plot_drawdown_periods(returns_pivot_df[id_min])
# plt.subplot(rows, 1, 8)
# pf.plot_drawdown_underwater(returns_pivot_df[id_min])


def t():
    program = Program()
    print(program.args)
    # wandbstats = Reload(program=program)
    # foo = wandbstats.download_test_history()
    # foo = wandbstats.initialize_stats()
    return None


if __name__ == "__main__":
    t()
