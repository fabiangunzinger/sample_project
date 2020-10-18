import pandas as pd
from .counter import counter, add_count


@counter
def min_number_of_months(df, min_months=6):
    """At least 6 months of data."""
    def helper(g):
        num_months = g.transaction_date.dt.to_period('M').nunique()
        return num_months > min_months
    return df.groupby('user_id').filter(helper)


@counter
def current_account(df):
    """At least one current account."""
    def helper(g):
        return 'current' in g.account_type.str.lower().unique()
    return df.groupby('user_id').filter(helper)


@counter
def min_txns_and_spend(df, min_txns=5, min_spend=200):
    """At least 5 transactions and spend of GBP200 per month."""
    def helper(g):
        txns = g.resample('M', on='transaction_date').transaction_id.size()
        debits = g[g.amount > 0]
        spend = debits.resample('M', on='transaction_date').amount.sum()
        return txns[1:-1].min() >= min_txns and spend[1:-1].min() >= min_spend
    return df.groupby('user_id').filter(helper)


@counter
def income_pmts(df):
    """Income payments in 2/3 of all observed months."""
    def helper(g):
        tot_months = g.ym.nunique()
        inc_months = g[g.tag.str.contains('_income')].ym.nunique()
        return (inc_months / tot_months) > (2/3)
    data = df[['user_id', 'transaction_date', 'tag']].copy()
    data['ym'] = data.transaction_date.dt.to_period('M')
    usrs = data.groupby('user_id').filter(helper).user_id.unique()
    return df[df.user_id.isin(usrs)]


@counter
def income_amount(df, lower=5_000, upper=100_000):
    """Yearly incomes between 5k and 100k.

    Yearly income calculated on rolling basis from first month of data,
    last year excluded as it has probably incomplete data.
    """
    def helper(g):
        first_month = g.transaction_date.min().strftime('%b')
        yearly_freq = 'AS-' + first_month.upper()
        year = pd.Grouper(freq=yearly_freq, key='transaction_date')
        yearly_inc = (g[g.tag.str.contains('_income')]
                      .groupby(year)
                      .amount.sum().mul(-1))
        return yearly_inc[:-1].between(lower, upper).all()

    return df.groupby('user_id').filter(helper)


@counter
def max_accounts(df):
    """No more than 10 active accounts in any year."""
    year = pd.Grouper(freq='M', key='transaction_date')
    usr_max = (df.groupby(['user_id', year]).account_id.nunique()
               .groupby('user_id').max())
    users = usr_max[usr_max <= 10].index
    return df[df.user_id.isin(users)]


@counter
def max_debits(df):
    """Debits of no more than 100k in any month."""
    month = pd.Grouper(freq='M', key='transaction_date')
    debits = df[df.amount > 0]
    usr_max = (debits.groupby(['user_id', month]).amount.sum()
               .groupby('user_id').max())
    users = usr_max[usr_max <= 100_000].index
    return df[df.user_id.isin(users)]


@counter
def working_age(df):
    """Working-age."""
    age = 2020 - df.year_of_birth
    return df[age.between(18, 64)]


def select_sample(df):
    return (
        df
        .pipe(add_count, 'Raw sample')
        .pipe(min_number_of_months)
        .pipe(current_account)
        .pipe(min_txns_and_spend)
        .pipe(income_pmts)
        .pipe(income_amount)
        .pipe(max_accounts)
        .pipe(max_debits)
        .pipe(working_age)
        .pipe(add_count, 'Final sample')
    )
