# -*- coding: utf-8 -*-
##
import sys


##
sys.path.append("./src/")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

##
import tqdm
import fundamentalanalysis as fa

##
from rl.data.Company import Company


def download(symbols: list, api_key: str, period: str = "quarter") -> dict:
    all_symbols_info = {}
    pbar = tqdm.tqdm(list(symbols))
    for symbol in pbar:
        ##
        pbar.set_description("Symbol: %s" % symbol)

        ##
        # Download Data
        # dcf = fa.discounted_cash_flow(symbol, api_key, period=period)
        balance_sheet = fa.balance_sheet_statement(symbol, api_key, period=period)
        income_statement = fa.income_statement(symbol, api_key, period=period)
        cash_flow_statement = fa.cash_flow_statement(symbol, api_key, period=period)
        key_metrics = fa.key_metrics(symbol, api_key, period=period)
        financial_ratios = fa.financial_ratios(symbol, api_key, period=period)
        growth = fa.financial_statement_growth(symbol, api_key, period=period)

        ##
        # Add data
        all_symbols_info[symbol] = Company(
            symbol=symbol,
            # dcf=dcf,
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            cash_flow_statement=cash_flow_statement,
            key_metrics=key_metrics,
            financial_ratios=financial_ratios,
            growth=growth,
        )

    return all_symbols_info
