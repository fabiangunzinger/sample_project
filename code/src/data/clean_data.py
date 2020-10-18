import numpy as np
import pandas as pd
from src import config


def drop_last_month(df):
    """Drop last month, which might have missing data.
    For first month, Jan 2012, we have complete data.
    """
    ym = df.transaction_date.dt.to_period('M')
    return df[ym < ym.max()]


def clean_categoricals(df: pd.DataFrame):
    """Strip categorical values and convert to lowercase."""
    def helper(col):
        return col.astype(str).str.lower().str.strip().astype('category')
    df = df.copy()
    cols = df.select_dtypes('category')
    df[cols.columns] = cols.apply(helper)
    return df


def clean_tags(df):
    """Replace parenthesis with dash for save regex searches."""
    df = df.copy()
    for tag in ['up_tag', 'auto_tag', 'manual_tag']:
        df[tag] = df[tag].str.replace('(', '- ').str.replace(')', '')
    return df


def clean_gender(df):
    """Categorise 'u' as missing."""
    df = df.copy()
    transformed = df.gender.astype('str').replace('u', None)
    df['gender'] = transformed.astype('category')
    return df


def order_salaries(df):
    """Turn salary range into ordered variable."""
    df = df.copy()
    cats = ['< 10k', '10k to 20k', '20k to 30k',
            '30k to 40k', '40k to 50k', '50k to 60k',
            '60k to 70k', '70k to 80k', '> 80k']
    df['salary_range'] = (df.salary_range.cat
                          .set_categories(cats, ordered=True))
    return df


def drop_duplicate_accounts(df):
    pass


def drop_business_accounts(df):
    pass


def correct_up_tag(df):
    """Make up_tag equal manual tag if it exists and auto_tag otherwise.
    This is how auto tag is supposed to behave but doesn't always.
    """
    df = df.copy()
    df['up_tag'] = np.where(df.manual_tag.ne('no tag'),
                            df.manual_tag, df.auto_tag)
    return df


def add_tag(df):
    """Create empty corrected tag variable."""
    df = df.copy()
    df['tag'] = None
    return df


def tag_pmt_pairs(df, knn=5):
    """Tag payments from one account to another as transfers.

    Identification criteria:
    1. same user
    2. larger than GBP50
    3. same amount
    4. no more than 2 days apart
    5. of the opposite sign (debit/credit)
    6. not already part of another transfer pair. This can happen in two ways:
       - A txn forms a pair with two neighbours at different distances,
         addressed in <1>.
       - A txn forms a pair with a neighbour and the neighbour with one of its
         own neighbours, addressed in <2>.

    Code sorts data by user, amount, and transaction date, and checks for each
    txn and each of its k nearest preceeding neighbours whether, together, they
    meet the above criteria.
    """
    df = df.copy()
    df['amount'] = df.amount.abs()
    df = df.sort_values(['user_id', 'amount', 'transaction_date'])
    for k in range(1, knn+1):
        is_tfr = (
            (df.user_id == df.user_id.shift(k))
            & (df.amount > 50)
            & (df.amount == df.amount.shift(k))
            & (df.transaction_date.diff(k).dt.days <= 2)
            & (df.credit_debit != df.credit_debit.shift(k))
            & (df.tag.ne('transfers'))                       # <1>
            & (df.tag.shift(k).ne('transfers'))              # <1>
        )
        # tag first txn of pair
        mask = is_tfr.eq(True) & is_tfr.shift(k).eq(False)   # <2>
        df['tag'] = np.where(mask, 'transfers', df.tag)
        # tag second txn of pair
        mask = is_tfr.shift(-k).eq(True)
        df['tag'] = np.where(mask, 'transfers', df.tag)
    return df


def tag_tranfsers(df):
    """Tag txns with description indicating tranfser payment."""
    df = df.copy()
    tfr_strings = [' ft', ' trf', 'xfer', 'transfer']
    exclude = ['fee', 'interest']
    mask = (df.transaction_description.str.contains('|'.join(tfr_strings))
            & ~df.transaction_description.str.contains('|'.join(exclude)))
    df.loc[mask, 'tag'] = 'transfers'
    return df


def drop_untagged(df):
    """Drop untagged transactions."""
    mask = (df.up_tag.eq('no tag')
            & (df.manual_tag.eq('no tag'))
            & (df.auto_tag.eq('no tag')))
    return df[~mask]


def tag_incomes(df):
    """Tag earnings, pensions, benefits, and other income.
    Based on Appendix A in Haciouglu et al. (2020).
    """
    df = df.copy()
    incomes = {
        'earnings': [
            'salary or wages - main',
            'salary or wages - other',
            'salary - secondary',
        ],
        'pensions': [
            'pension - other',
            'pension',
            'work pension',
            'state pension',
            'pension or investments',
        ],
        'benefits': [
            'benefits',
            'family benefits',
            'job seekers benefits',
            'other benefits',
            'incapacity benefits'
        ],
        'other': [
            'rental income - whole property',
            'rental income - room',
            'rental income',
            'irregular income or gifts',
            'miscellaneous income - other',
            'investment income - other',
            'loan or credit income',
            'bond income',
            'interest income',
            'dividend',
        ],
    }
    for type, tags in incomes.items():
        mask = (df[config.TAGVAR].str.match('|'.join(tags))
                & df.credit_debit.eq('credit'))
        df.loc[mask, 'tag'] = type + '_income'
    return df


def fill_tag(df):
    """Replace tag with auto tag if missing."""
    df = df.copy()
    df['tag'] = df.tag.where(df.tag.notna(), df.auto_tag).astype('category')
    return df


def drop_card_repayments(df):
    """Drop card repayment transactions from current accounts."""
    tags = ['credit card repayment', 'credit card payment', 'credit card']
    mask = (df.auto_tag.str.contains('|'.join(tags))
            & df.account_type.eq('current'))
    return df[~mask]


def sign_amount(df):
    """Make credits negative."""
    df = df.copy()
    credit = df.credit_debit.eq('credit')
    df['amount'] = np.where(credit, df.amount.mul(-1), df.amount)
    return df


def reorder_columns(df):
    first = [
        'user_id', 'transaction_date', 'amount',
        'transaction_description', 'merchant_name',
        'auto_tag', 'tag', 'manual_tag'
    ]
    rest = set(df.columns) - set(first)
    ordered = first + list(rest)
    return df[ordered]


def drop_unneeded_columns(df):
    cols = [
        'msoa',
        'lsoa',
        'data_warehouse_date_created',
        'data_warehouse_date_last_updated',
        'transaction_updated_flag',
    ]
    return df.drop(cols, axis=1)


def sort_rows(df):
    return df.sort_values(['user_id', 'transaction_date'], ignore_index=True)


def clean_data(df):
    """Perform simple cleaning operations required for selection."""
    return (
        df
        .pipe(drop_last_month)
        .pipe(clean_categoricals)
        .pipe(clean_tags)
        .pipe(clean_gender)
        .pipe(order_salaries)
        .pipe(correct_up_tag)
        .pipe(add_tag)
        .pipe(tag_pmt_pairs)
        .pipe(tag_tranfsers)
        .pipe(tag_incomes)
        .pipe(fill_tag)
        .pipe(drop_untagged)
        .pipe(sign_amount)
        .pipe(reorder_columns)
        .pipe(drop_unneeded_columns)
        .pipe(sort_rows)
    )
