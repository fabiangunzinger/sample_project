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


def haciouglu_period(df):
    """Restrict sample to Jan 2019 - Jun 2020."""
    return (df.set_index('transaction_date').loc['2019-01':'2020-06']
            .reset_index())


def min_txns_and_spend(df):
    def helper(g):
        txns = g.resample('M', on='transaction_date').transaction_id.size()
        debits = g[g.amount > 0]
        spend = debits.resample('M', on='transaction_date').amount.sum()
        return (txns.min() >= 5) & (spend.min() >= 200)
    return df.groupby('user_id').filter(helper)


def income(df):
    def helper(g):
        g = g[g.tag.str.match('|'.join(tags))]
        data2019 = g[g.transaction_date.dt.year.eq(2019)]
        data2020 = g[g.transaction_date.dt.year.eq(2020)]
        return (
            (data2019.transaction_date.dt.month.nunique() >= 8)
            & (data2020.transaction_date.dt.month.nunique() >= 2)
            & (data2019[data2019.amount < 0].amount.sum() <= -5000)
            & (data2019[data2019.amount < 0].amount.sum() > -60000)
        )
    tags = ['earnings', 'pensions', 'benefits', 'other income']
    return df.groupby('user_id').filter(helper)


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


def select_sample_hacioglu(df):
    return (
        df
        .pipe(haciouglu_period)
        .pipe(current_account)
        .pipe(min_txns_and_spend)
        # .pipe(june_refresh)
        # .pipe(duplicate_accounts)
        # .pipe(business_accounts)
        # .pipe(min_expenditure)
        .pipe(income)

    )
