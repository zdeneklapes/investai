# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
from plotly import graph_objs as go

from examples.portfolio_allocation_fa_dji30.PortfolioAllocationEnv import PortfolioAllocationEnv
from examples.portfolio_allocation_fa_dji30.dataset import StockDataset
from extra.math.finance.minimum_variance import minimum_variance
from model_config.plot import get_baseline, get_daily_return, convert_daily_return_to_pyfolio_ts
from project_configs.program import Program


class Test:
    def __init__(self, program: Program, stock_dataset: StockDataset):
        self.program: Program = program
        self.stock_dataset: StockDataset = stock_dataset
        self.env: PortfolioAllocationEnv = PortfolioAllocationEnv(
            df=self.stock_dataset.test_dataset,
            initial_portfolio_value=100_000,
            tickers=self.stock_dataset.tickers,
            features=self.stock_dataset.get_features(),
            save_path=self.program.experiment_dir.algo,
            start_from_index=self.stock_dataset.test_dataset.index[0]
        )

    def test(self, model_path: Path = None) -> None:
        if model_path is None:  # Test all
            for model_path in self.program.experiment_dir.models.iterdir():
                self.program.experiment_dir.set_algo(model_path.name)
                self.test_model(model_path)
        else:  # Test one
            self.test_model(model_path)

    def test_model(self, model_path: Path) -> None:
        # Get files
        model_files = [
            m for m in model_path.iterdir()
            if m.is_file() and m.name.startswith('model') and m.suffix == ".zip"
        ]
        model_files.sort(key=lambda x: int(x.name.split("_")[1]))  # Sort by number of steps

        # Prepare model
        algorithm_name = model_path.name.split("_")[0]
        # algorithm = Algorithm(
        #     program=self.program,
        #     algorithm=algorithm_name,
        #     env=self.env
        # )
        # model = algorithm.get_model(tensorboard_log=self.program.experiment_dir.tensorboard.as_posix(), verbose=0)

        # All checkpoints
        # pbar = tqdm(model_files)
        # for checkpoint_model_path in pbar:
        #     pbar.set_description(f"Testing {algorithm_name} model: {checkpoint_model_path.name}")
        #     self.get_checkpoints_performance(model, checkpoint_model_path)

    def get_checkpoints_performance(self, model, checkpoint_model_path: Path) -> None:
        loaded_model = model.load(checkpoint_model_path.as_posix())
        obs = self.env.reset()
        done = False
        while not done:
            action, _states = loaded_model.predict(obs)
            done = self.env.step(action)[2]

        # Save memory
        memory_name = f"model_{checkpoint_model_path.name.split('_')[1]}_steps_memory.json"
        self.env._memory.save(self.program.experiment_dir.algo.joinpath(memory_name))

    def get_memory_stats(self, memory_file: Path) -> None:
        pass

    def plot_stats(self) -> None:
        import pyfolio
        for p in self.program.experiment_dir.models.iterdir():
            memories = [m for m in p.iterdir() if m.name.endswith("memory.json")]
            memories.sort(key=lambda x: int(x.name.split("_")[1]))
            # TODO: Take only best memory

            for memory in memories:
                df = pd.read_json(memory)
                # Get Baseline
                baseline_df = get_baseline(ticker='^DJI', start=df.loc[0, 'date'], end=df.tail(1).iloc[0]['date'])
                baseline_returns = get_daily_return(baseline_df, value_col_name="close")

                # Get DRL
                df_ts = convert_daily_return_to_pyfolio_ts(df, "portfolio_return")

                with pyfolio.plotting.plotting_context(font_scale=1.1):
                    pyfolio.create_full_tear_sheet(returns=df_ts, benchmark_rets=baseline_returns, set_context=False)

                # print("\n\n\n\n")
                # print(memory.name)
                break

    def plot_compare_portfolios(self):
        for p in self.program.experiment_dir.models.iterdir():
            memories = [m for m in p.iterdir() if m.name.endswith("memory.json")]
            memories.sort(key=lambda x: int(x.name.split("_")[1]))
            # TODO: Take only best memory
            for memory in memories:
                #
                df = pd.read_json(memory)

                #
                baseline_df = get_baseline(ticker='^DJI', start=df.loc[0, 'date'], end=df.tail(1).iloc[0]['date'])
                baseline_returns = get_daily_return(baseline_df, value_col_name="close")

                #
                model_cumpod = (1 + df['portfolio_return']).cumprod() - 1
                dji_cumpod = (1 + baseline_returns).cumprod() - 1
                min_var_cumpod = minimum_variance(self.stock_dataset.test_dataset)

                #
                trace0_portfolio = go.Scatter(x=df['date'], y=model_cumpod, mode='lines',
                                              name=f"{p.name}]")
                trace1_portfolio = go.Scatter(x=df['date'], y=dji_cumpod, mode='lines', name='DJIA')
                trace2_portfolio = go.Scatter(x=df['date'], y=min_var_cumpod, mode='lines', name='Min-Variance')

                #
                fig = go.Figure()
                fig.add_trace(trace0_portfolio)
                fig.add_trace(trace1_portfolio)
                fig.add_trace(trace2_portfolio)

                #
                fig.update_layout(
                    legend=dict(
                        x=0,
                        y=1,
                        traceorder="normal",
                        font=dict(
                            family="sans-serif",
                            size=15,
                            color="black"
                        ),
                        bgcolor="White",
                        bordercolor="white",
                        borderwidth=2

                    ),
                )
                # fig.update_layout(legend_orientation="h")
                fig.update_layout(title={
                    # 'text': "Cumulative Return using FinRL",
                    'y': 0.85,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
                # with Transaction cost
                # fig.update_layout(title =  'Quarterly Trade Date')
                fig.update_layout(
                    #    margin=dict(l=20, r=20, t=20, b=20),

                    paper_bgcolor='rgba(1,1,0,0)',
                    plot_bgcolor='rgba(1, 1, 0, 0)',
                    # xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    xaxis={'type': 'date',
                           'tick0': df.loc[0, 'date'],
                           'tickmode': 'linear',
                           'dtick': 86400000.0 * 80}

                )
                fig.update_xaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1,
                                 gridcolor='LightSteelBlue', mirror=True)
                fig.update_yaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1,
                                 gridcolor='LightSteelBlue', mirror=True)
                fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

                fig.show()
                break
