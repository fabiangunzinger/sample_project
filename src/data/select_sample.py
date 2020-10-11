import pandas as pd
from .counter import counter, add_count


@counter
def min_number_of_months(df, thresh=6):
    """Keep users we observe for at least 6 months."""
    def helper(g):
        num_months = g.transaction_date.dt.to_period('M').nunique()
        return num_months > thresh
    return df.groupby('user_id').filter(helper)


@counter
def current_account(df):
    """Keep users with at least one current account."""
    def helper(g):
        return 'current' in g.account_type.str.lower().unique()
    return df.groupby('user_id').filter(helper)


@counter
def min_number_transactions(df, min_txns=12, prop=0.8):
    """Keep users with a minumum number of txs in most months.
    Most months is defined as prop of all observed months.
    """
    def helper(g):
        monthly_txs = g.resample('M', on='transaction_date').size()
        num_mths = len(monthly_txs)
        num_mths_min_txs = (monthly_txs > min_txns).sum()
        return (num_mths_min_txs / num_mths) > prop
    return df.groupby('user_id').filter(helper)


@counter
def diverse_spending(df, min_tags=1, prop=0.8):
    """Keep users who spend money on more than one category most months.
    Most months is defined as prop of all observed months.
    """
    def helper(g):
        monthly_tgs = g.resample('M', on='transaction_date').auto_tag.nunique()
        num_mths = len(monthly_tgs)
        num_mths_min_tgs = (monthly_tgs > min_tags).sum()
        return (num_mths_min_tgs / num_mths) > prop
    return df.groupby('user_id').filter(helper)


@counter
def working_age(df):
    """Keep working-age users only."""
    age = 2020 - df.year_of_birth
    return df[age.between(18, 65)]


def select_sample(df):
    return (
        df
        .pipe(add_count)
        .pipe(min_number_of_months)
        .pipe(current_account)
        .pipe(min_number_transactions, min_txns=5, prop=1)
        .pipe(diverse_spending)
        .pipe(working_age)
        .pipe(add_count, 'end')
    )
