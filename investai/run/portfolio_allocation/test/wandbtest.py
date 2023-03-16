# -*- coding: utf-8 -*-
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shared.program import Program
from run.shared.tickers import DOW_30_TICKER
from run.shared.algorithmsb3 import ALGORITHM_SB3_TYPE
from run.portfolio_allocation.dataset.stockfadailydataset import StockFaDailyDataset
from run.portfolio_allocation.envs.portfolioallocationenv import PortfolioAllocationEnv


class WandbTest:
    def __init__(self, program: Program, dataset: StockFaDailyDataset):
        self.program: Program = program
        self.dataset: StockFaDailyDataset = dataset

    def _init_environment(self):
        self.program.log.info("Init environment")
        env = PortfolioAllocationEnv(df=self.dataset.test_dataset, tickers=self.dataset.tickers,
                                     features=self.dataset.get_features(),
                                     start_data_from_index=self.program.args.start_data_from_index)
        env = Monitor(
            env,
            Path(self.program.project_structure.wandb).as_posix(),
            allow_early_resets=True)  # stable_baselines3.common.monitor.Monitor
        env = DummyVecEnv([lambda: env])
        return env

    def _deinit_environment(self, env):
        self.program.log.info("Deinit environment")
        env.close()

    def test(self, model: ALGORITHM_SB3_TYPE, deterministic=True) -> None:
        environment = self._init_environment()
        obs = environment.reset()
        # -2 because we don't want to go till terminal state, because the environment will be reset
        for i in range(len(environment.envs[0].env._df.index.unique()) - 2):
            action, _ = model.predict(obs, deterministic=deterministic)
            environment.step(action)

    # def test_model(self, model_path: Path) -> None:
    #     # Get files
    #     model_files = [
    #         m for m in model_path.iterdir()
    #         if m.is_file() and m.name.startswith('model') and m.suffix == ".zip"
    #     ]
    #     model_files.sort(key=lambda x: int(x.name.split("_")[1]))  # Sort by number of steps
    #
    #     # Prepare model
    #     # algorithm_name = model_path.name.split("_")[0]
    #     # algorithm = Algorithm(
    #     #     program=self.program,
    #     #     algorithm=algorithm_name,
    #     #     env=self.env
    #     # )
    #     # model = algorithm.get_model(tensorboard_log=self.program.experiment_dir.tensorboard.as_posix(), verbose=0)
    #
    #     # All checkpoints
    #     # pbar = tqdm(model_files)
    #     # for checkpoint_model_path in pbar:
    #     #     pbar.set_description(f"Testing {algorithm_name} model: {checkpoint_model_path.name}")
    #     #     self.get_checkpoints_performance(model, checkpoint_model_path)
    #
    # def get_checkpoints_performance(self, model, checkpoint_model_path: Path) -> None:
    #     loaded_model = model.load(checkpoint_model_path.as_posix())
    #     obs = self.env.reset()
    #     done = False
    #     while not done:
    #         action, _states = loaded_model.predict(obs)
    #         done = self.env.step(action)[2]
    #
    #     # Save memory
    #     memory_name = f"model_{checkpoint_model_path.name.split('_')[1]}_steps_memory.json"
    #     self.env._memory.save_memory(self.program.project_structure.algo.joinpath(memory_name))

    # def get_memory_stats(self, memory_file: Path) -> None:
    #     pass
    #
    # def plot_stats(self) -> None:
    #     import pyfolio
    #     for p in self.program.experiment_dir.models.iterdir():
    #         memories = [m for m in p.iterdir() if m.name.endswith("memory.json")]
    #         memories.sort(key=lambda x: int(x.name.split("_")[1]))
    #         # TODO: Take only best memory
    #
    #         for memory in memories:
    #             df = pd.read_json(memory)
    #             # Get Baseline
    #             baseline_df = get_baseline(ticker='^DJI', start=df.loc[0, 'date'], end=df.tail(1).iloc[0]['date'])
    #             baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    #
    #             # Get DRL
    #             df_ts = convert_daily_return_to_pyfolio_ts(df, "portfolio_return")
    #
    #             with pyfolio.plotting.plotting_context(font_scale=1.1):
    #                 pyfolio.create_full_tear_sheet(returns=df_ts, benchmark_rets=baseline_returns, set_context=False)
    #
    #             # print("\n\n\n\n")
    #             # print(memory.name)
    #             break
    #
    # def plot_compare_portfolios(self):
    #     for p in self.program.models.iterdir():
    #         memories = [m for m in p.iterdir() if m.name.endswith("memory.json")]
    #         memories.sort(key=lambda x: int(x.name.split("_")[1]))
    #         # TODO: Take only best memory
    #         for memory in memories:
    #             #
    #             df = pd.read_json(memory)
    #
    #             #
    #             baseline_df = get_baseline(ticker='^DJI', start=df.loc[0, 'date'], end=df.tail(1).iloc[0]['date'])
    #             baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    #
    #             #
    #             model_cumpod = (1 + df['portfolio_return']).cumprod() - 1
    #             dji_cumpod = (1 + baseline_returns).cumprod() - 1
    #             min_var_cumpod = minimum_variance(self.dataset.test_dataset)
    #
    #             #
    #             trace0_portfolio = go.Scatter(x=df['date'], y=model_cumpod, mode='lines',
    #                                           name=f"{p.name}]")
    #             trace1_portfolio = go.Scatter(x=df['date'], y=dji_cumpod, mode='lines', name='DJIA')
    #             trace2_portfolio = go.Scatter(x=df['date'], y=min_var_cumpod, mode='lines', name='Min-Variance')
    #
    #             #
    #             fig = go.Figure()
    #             fig.add_trace(trace0_portfolio)
    #             fig.add_trace(trace1_portfolio)
    #             fig.add_trace(trace2_portfolio)
    #
    #             #
    #             fig.update_layout(
    #                 legend=dict(
    #                     x=0,
    #                     y=1,
    #                     traceorder="normal",
    #                     font=dict(
    #                         family="sans-serif",
    #                         size=15,
    #                         color="black"
    #                     ),
    #                     bgcolor="White",
    #                     bordercolor="white",
    #                     borderwidth=2
    #
    #                 ),
    #             )
    #             # fig.update_layout(legend_orientation="h")
    #             fig.update_layout(title={
    #                 # 'text': "Cumulative Return using FinRL",
    #                 'y': 0.85,
    #                 'x': 0.5,
    #                 'xanchor': 'center',
    #                 'yanchor': 'top'})
    #             # with Transaction cost
    #             # fig.update_layout(title =  'Quarterly Trade Date')
    #             fig.update_layout(
    #                 #    margin=dict(l=20, r=20, t=20, b=20),
    #
    #                 paper_bgcolor='rgba(1,1,0,0)',
    #                 plot_bgcolor='rgba(1, 1, 0, 0)',
    #                 # xaxis_title="Date",
    #                 yaxis_title="Cumulative Return",
    #                 xaxis={'type': 'date',
    #                        'tick0': df.loc[0, 'date'],
    #                        'tickmode': 'linear',
    #                        'dtick': 86400000.0 * 80}
    #
    #             )
    #             fig.update_xaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1,
    #                              gridcolor='LightSteelBlue', mirror=True)
    #             fig.update_yaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1,
    #                              gridcolor='LightSteelBlue', mirror=True)
    #             fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')
    #
    #             fig.show()
    #             break


# def get_wandb_runs() -> pd.DataFrame:
#     api = wandb.Api()
#     entity, project = "zlapik", "ai-investing"  # set to your entity and project
#     runs = api.runs(entity + "/" + project)
#
#     summary_list, config_list, name_list = [], [], []
#     for run in runs:
#         # .summary contains the output keys/values for metrics like accuracy.
#         #  We call ._json_dict to omit large files
#         summary_list.append(run.summary._json_dict)
#
#         # .config contains the hyperparameters.
#         #  We remove special values that start with _.
#         config_list.append(
#             {k: v for k, v in run.config.items()
#              if not k.startswith('_')})
#
#         # .name is the human-readable name of the run.
#         name_list.append(run.name)
#
#     runs_df = pd.DataFrame({
#         "summary": summary_list,
#         "config": config_list,
#         "name": name_list
#     })
#     return runs_df


def main():
    from dotenv import load_dotenv

    program = Program()
    load_dotenv(dotenv_path=program.project_structure.root.as_posix())
    dataset = StockFaDailyDataset(program, DOW_30_TICKER, program.args.dataset_split_coef)
    dataset.load_dataset(program.args.dataset_path)
    # t = WandbTest(program, dataset)  # noqa
    # t.test()
    # runs_df = get_wandb_runs()  # noqa


if __name__ == '__main__':
    main()
