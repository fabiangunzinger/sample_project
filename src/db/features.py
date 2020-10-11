import pandas as pd
from .registries import feature, preproc


@preproc
@feature
def pct_manual_tags(df):
    """Return proportion of tags manually set by user."""
    def helper(g):
        num_trans = g.amount.count()
        num_manual_tags = g.manual_tag.str.match('(?!no tag)').sum()
        return num_manual_tags / num_trans

    return df.groupby('user_id').apply(helper).rename('pct_manual_tags')


@preproc
@feature
def pct_credit(df):
    """Percentage of purchases financed by credit card."""
    def helper(g):
        mask = g.credit_debit.eq('debit')
        total_trans = g[mask].amount.count()
        credit_trans = g[mask].account_type.eq('credit card').sum()
        return credit_trans / total_trans

    return df.groupby('user_id').apply(helper).rename('pct_credit')


@preproc
@feature
def entropy(df):
    """Return Shannon Entropy for purchases of each user."""
    from scipy.stats import entropy
    mask = df.credit_debit.eq('debit')
    df = df[mask]
    num_cats = df.auto_tag.nunique()

    def calc_entropy(user, num_cats):
        total_purchases = len(user)
        cat_purchases = user.groupby('auto_tag').size()
        probs = (cat_purchases + 1) / (total_purchases + num_cats)
        return entropy(probs, base=2)

    g = df.groupby('user_id')
    return g.apply(calc_entropy, num_cats).rename('entropy')


@feature
def merchant_dummies(df):
    """Indicate whether user made purchase from merchant."""
    mask = df.credit_debit.eq('debit')
    df = df[mask].copy()
    df['merchant_name'] = (df.merchant_name
                           .str.replace('\W', '')
                           .str.replace(' ', '_')
                           .str[:10])

    def helper(g):
        return ((g.groupby('merchant_name').amount.sum() > 0)
                .astype(int)
                .add_prefix('merch_'))

    return df.groupby('user_id').apply(helper).unstack().fillna(0)


@feature
def tag_dummies(df):
    """Indicate whether user made purchase classified by tag."""
    mask = df.credit_debit.eq('debit')
    df = df[mask].copy()

    df['auto_tag'] = (df.auto_tag
                      .str.replace('\W', '')
                      .str.replace(' ', '_')
                      .str[:10])

    def helper(g):
        return ((g.groupby('auto_tag').amount.sum() > 0)
                .astype(int)
                .add_prefix('tag_'))

    return df.groupby('user_id').apply(helper).unstack().fillna(0)


@feature
def merchant_spending_shares(df):
    """Return spending shares by merchant."""
    mask = df.credit_debit.eq('debit')
    df = df[mask].copy()

    df['merchant_name'] = (df.merchant_name
                           .str.replace('\W', '')
                           .str.replace(' ', '_')
                           .str[:10])

    def helper(g):
        merch_spending = g.groupby('merchant_name').amount.sum()
        total_spending = g.amount.sum()
        shares = merch_spending / total_spending
        return shares.add_prefix('merchshare_')
    return df.groupby('user_id').apply(helper).unstack().fillna(0)


@feature
def tag_spending_shares(df):
    """Return spending shares by tag."""
    mask = df.credit_debit.eq('debit')
    df = df[mask].copy()

    df['auto_tag'] = (df.auto_tag
                      .str.replace('\W', '')
                      .str.replace(' ', '_'))

    def helper(g):
        tag_spending = g.groupby('auto_tag').amount.sum()
        total_spending = g.amount.sum()
        shares = tag_spending / total_spending
        return shares.add_prefix('tagshare_')
    return df.groupby('user_id').apply(helper).unstack().fillna(0)


@preproc
@feature
def grocery_shop_freq(df):
    """Return number of grocery shops per week."""
    def helper(g):
        num_days = g.transaction_date.max() - g.transaction_date.min()
        num_weeks = num_days / pd.Timedelta('1W')
        mask = g.auto_tag.eq('food, groceries, household')
        num_shops = g[mask].transaction_id.nunique()
        return num_shops / num_weeks
    return (
        df.groupby('user_id')
        .apply(helper)
        .rename('grocery_shop_freq')
    )


def tag_merchant_dummies(df):
    """Return interaction between tag and merchant dummies."""
    tags = df.filter(regex='^tag')
    merchs = df.filter(regex='^merch')
    data = pd.DataFrame()

    for t in tags:
        for m in merchs:
            data['tm_' + t[4:] + '_' + m[6:]] = tags[t] * merchs[m]
    return data
