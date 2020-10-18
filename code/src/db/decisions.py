import re
import numpy as np
import pandas as pd
from collections import Counter
from .registries import decision


def active_months(df):
    """Keep user-month observations with at least 12 txs."""
    mnths = df.transaction_date.dt.to_period('M')
    g = df.groupby(['user_id', mnths])
    mask = g.transaction_id.transform('count') > 11
    return df[mask]


@decision
def amazon_per_wk(df):
    """Return number of amazon purchases per week."""
    def helper(g):
        num_wks = g.transaction_date.dt.to_period('W').nunique()
        num_amzn = g.transaction_description.str.contains('amazon|amzn').sum()
        return num_amzn / num_wks
    df = active_months(df)
    g = df.groupby('user_id')
    return g.apply(helper).rename('amazon_per_wk').reset_index()


@decision
def groceries_per_wk(df):
    """Return number of grocery shops per week."""
    def helper(g):
        num_wks = g.transaction_date.dt.to_period('W').nunique()
        num_shops = g.auto_tag.str.contains('groceries').sum()
        return num_shops / num_wks
    df = active_months(df)
    g = df.groupby('user_id')
    return g.apply(helper).rename('groceries_per_wk').reset_index()


@decision
def meals_per_wk(df):
    """Return number of meals out per week."""
    def h(g):
        num_wks = g.transaction_date.dt.to_period('W').nunique()
        regex = 'dining or going out|lunch or snacks'
        num_meals = g.auto_tag.str.contains(regex).sum()
        return num_meals / num_wks
    df = active_months(df)
    g = df.groupby('user_id')
    return g.apply(h).rename('meals_per_wk').reset_index()


# ==============================================================================
# o2 mobile phone payments


def make_o2_subset(df):
    """Keep observations used for o2 classification."""
    misclass = [
        'academy', 'apollo', 'arena', 'box office',
        'burger', 'byron', 'car park', 'cineworld',
        'coffee', 'dome', 'entertainment', 'five guys',
        'frankie & bennys', 'jimmys', 'parking',
        'the o2', 'wasabi', 'watermargin', 'zizzi',
    ]
    prepay = ['pay & go', 'prepay', 'top up']
    regex = re.compile('|'.join(misclass + prepay))
    tagsum = df[['tag', 'auto_tag', 'manual_tag']].sum(1)
    o2_trx = (
        (tagsum.str.contains('mobile'))
        & (df.merchant_name.eq('o2'))
        & (~df.transaction_description.str.contains(regex))
    )
    credits = df.amount > 0
    return df[o2_trx & credits]


def large_payment(df, thresh=100):
    """Identify potentil phone purchases."""
    df['large_payment'] = df.amount > thresh
    return df


def larger_than_neighbours(df, multiple=3, window=20):
    """Identify payments that are larger than their neighbours.

    Ensures temporary high airtime bills aren't mistaken for phone purchase.
    """
    def helper(g):
        center = g.amount.rolling(window, center=True).mean()
        right = g.amount.rolling(window).mean()
        left = g.amount.rolling(half_window).mean().shift(-(half_window-1))
        mean = g.amount.mean()
        rolling = center
        rolling = rolling.where(rolling.notna(), right)  # fill start
        rolling = rolling.where(rolling.notna(), left)   # fill end
        rolling = rolling.where(rolling.notna(), mean)   # if obs < window
        g['larger_than_neighbours'] = g.amount > rolling * multiple
        return g

    half_window = round(window/2)
    df = df.sort_values(['user_id', 'transaction_date'])
    return df.groupby('user_id').apply(helper)


def rare_large_payment(df, thresh=100):
    """Return true if user makes no more than one large payment per year.

    Ensures regular high airtime bills aren't classified as phone purchases.
    """
    def helper(g):
        num_years = g.transaction_date.dt.year.nunique()
        num_large_paym = (g.amount > thresh).sum()
        g['rare_large_payment'] = num_large_paym <= num_years
        return g

    return df.groupby('user_id').apply(helper)


def multiple_payments(df):
    """Return true if user made multiple o2 pmts on same day.

    Indicator for airtime and phone payments.
    """
    def helper(g):
        g['multiple_payments'] = len(g.index) > 1
        return g

    return df.groupby(['user_id', 'transaction_date']).apply(helper)


def long_series(df, min_length=6):
    """Tag users with large number of duplicate amounts."""
    data = list(zip(df.user_id, df.amount))

    # dict with user_id as key and amounts as list of values
    idx = {}
    for d in data:
        idx.setdefault(d[0], []).append(d[1])

    # dict with user_id as key and count of most occuring amount as value
    result = {}
    for user_id in idx:
        counts = Counter(idx[user_id]).values()
        result[user_id] = max(counts)

    # test condition and append to df
    s = pd.Series(result) > min_length
    s.name = 'long_series'
    s.index.name = 'user_id'
    return df.merge(s.reset_index(), validate='m:1')


def classify_o2(df):
    """Classify users as upfront or instalment payers.

    Users who meet both conditions are classified as missing.
    """
    upfront = (df.large_payment
               & df.rare_large_payment
               & df.larger_than_neighbours)
    instalments = df.long_series & df.multiple_payments
    df['o2_phone'] = 0
    df['o2_phone'] = np.where(upfront & ~instalments, 1, df.o2_phone)
    df['o2_phone'] = np.where(instalments & ~upfront, 2, df.o2_phone)

    def helper(g):
        sum = g.o2_phone.sum()
        if (sum == 0) | (sum == 3):
            g['o2_phone'] = None
            return g
        elif sum == 1:
            g['o2_phone'] = 1
            return g
        elif sum == 2:
            g['o2_phone'] = 0
            return g

    df = df[['user_id', 'o2_phone']].drop_duplicates()
    df = df.groupby('user_id').apply(helper)
    return df.drop_duplicates()


@decision
def o2_phone(df):
    """Create dummy indicating mode of payment for new phone."""
    return (
        df
        .pipe(make_o2_subset)
        .pipe(long_series)
        .pipe(large_payment)
        .pipe(rare_large_payment)
        .pipe(larger_than_neighbours)
        .pipe(multiple_payments)
        .pipe(classify_o2)
    )


# ==============================================================================
# car insurance payments


def make_carins_subset(df):
    """Keep observations used for car insurance classification."""
    tagsum = df[['tag', 'auto_tag', 'manual_tag']].sum(1)
    carins_trans = ((tagsum.str.contains('vehicle insurance'))
                    & (df.manual_tag.ne('home insurance')))
    credit = df.amount > 0
    return df[carins_trans & credit]


def payment_series(df, thresh=5):
    """Return true if user has series of payments of same amount.

    Default threshold series length is set to 5 based on
    data inspection: there is a substantial number of users
    who celarly seem to pay monthly but for whom the precise
    amount changes slightly after about 5 or 6 identical payments.

    Relies on equal rather than subsequent payments because
    subsequent payments are sensible to payments for mutliple
    cars (two parallel payment streaks on alternating dates),
    as well as to random/misclassified payments.
    """
    def helper(g):
        diff = g.amount.diff()
        longest = diff.cumsum().value_counts().max()
        g['series'] = longest >= thresh
        return g

    if df.empty:
        df['series'] = None
        return df
    df = df.sort_values(['user_id', 'amount'])
    return df.groupby('user_id').apply(helper)


def large_payments(df, thresh=250):
    """Return true if user made unique large payment.

    Unique large payments are payments that are not
    part of a series of payments of equal amount, to
    take into account that some users might make montlhy
    payments that are higher than the hlp_threshold.
    """
    def helper(g):
        g['large_paym'] = (g.amount > thresh).max()
        return g

    if df.empty:
        df['large_paym'] = None
        return df
    return df.groupby('user_id').apply(helper)


def classify_carins(df):
    """Classify users as monthly or yearly.

    Users who meet both conditions are classified as missing.
    """
    monthly = df.series
    yearly = df.large_paym
    df['carins_paym'] = None
    df['carins_paym'] = np.where(monthly & ~yearly, 0, df.carins_paym)
    df['carins_paym'] = np.where(yearly & ~monthly, 1, df.carins_paym)
    return df[['user_id', 'carins_paym']].drop_duplicates()


@decision
def carins_paym(df):
    """Create dummy indicating mode of payment for car insurance."""
    return (
        df
        .pipe(make_carins_subset)
        .pipe(payment_series)
        .pipe(large_payments)
        .pipe(classify_carins)
    )


# ==============================================================================
# insurer swaps


@decision
def insurer_swaps(df, thresh=1/3):
    """Classify user as frequent or infrequent insurer swapper.

    'Frequent' is defined as swapping home or car insurer at least thresh times
    per year; if thresh equals 0.5, a frequent swaper makes at least one swap
    every other year.
    """
    def helper(g):
        num_trans = g.auto_tag.str.match('(vehicle|home) insurance').sum()
        if num_trans == 0:
            return None

        years = g.transaction_date.dt.year.nunique()

        mask = g.auto_tag.str.match('home insurance')
        swaps = max(0, g[mask].merchant_name.nunique() - 1)
        mask = g.auto_tag.str.match('vehicle insurance')
        swaps += max(0, g[mask].merchant_name.nunique() - 1)

        return 1 if (swaps / years) > thresh else 0

    return (df.groupby('user_id')
            .apply(helper)
            .rename('insurer_swaps')
            .reset_index())
