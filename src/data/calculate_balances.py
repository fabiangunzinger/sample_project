def latest_balance_as_row(df):
    """Add latest balance as a temporary row."""
    data = df.copy()
    cols = ['account_id', 'account_last_refreshed', 'latest_balance']
    data = data[cols].drop_duplicates()

    data['transaction_description'] = '_balance'
    data.rename(columns={'account_last_refreshed':  'transaction_date',
                         'latest_balance': 'amount'}, inplace=True)

    return df.append(data).sort_values(['account_id', 'transaction_date'])


def calculate_balances(df, window=3, aggfunc='mean'):
    """
    Calculate daily balances.

    Default calculates balance for each day as the
    mean of the balances of the current day and
    the two subsequent days.
    """
    def helper(g):
        mask = g.transaction_description.eq('_balance')
        refresh_date = g[mask].transaction_date.dt.date.values[0]
        pre_refresh = g.transaction_date.dt.date < refresh_date
        latest_balance = g[mask].amount.values[0]

        pre_refresh_balances = (
            g[pre_refresh]
            .set_index('transaction_date')
            .resample('D').amount.sum()
            .sort_index(ascending=False)
            .rolling(window=window, min_periods=1).agg(aggfunc)
            .cumsum()
            .add(latest_balance)
        )
        post_refresh_balances = (
            g[~pre_refresh]
            .set_index('transaction_date')
            .resample('D').amount.sum().mul(-1)
            .sort_index(ascending=True)
            .rolling(window=window, min_periods=1).agg(aggfunc)
            .cumsum()
        )
        return (
            pre_refresh_balances
            .append(post_refresh_balances)
            .sort_index()
            .rename('balance')
        )

    data = latest_balance_as_row(df)
    balances = data.groupby('account_id').apply(helper).reset_index()

    return df.merge(balances, how='left', validate='m:1')
