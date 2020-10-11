import pandas as pd


def selection_table(dict):
    """Create sample selection table for data appendix."""
    desc = {
        'start': 'Original sample',
        'min_number_of_months': 'and observed for at least six months',
        'current_account': 'and added a current account',
        'min_number_transactions': 'and at least five transactions most months',
        'diverse_spending': 'and spending on more than one category most months',
        'working_age': 'and between 18 and 65 years old',
        'end': 'Final sample'
    }
    df = pd.DataFrame(dict.items(), columns=['step', 'counts'])

    # reshape
    df[['step', 'metric']] = df.step.str.split('-', expand=True)
    df = (df.groupby(['step', 'metric'], sort=False)
          .counts.sum()
          .unstack('metric')
          .rename_axis(columns=None)
          .reset_index())

    # format
    df['step'] = df['step'].map(desc)
    ints = ['users', 'acc', 'txs']
    df[ints] = df[ints].applymap('{:,.0f}'.format)
    floats = ['value']
    df[floats] = df[floats].applymap('{:,.1f}'.format)

    # add Latex padding to middle rows
    indent = df[1:-1].index
    df.loc[indent, 'step'] = '\quad ' + df.step

    # rename colums
    df.columns = [
        '',
        'Users',
        'Accounts',
        'Transactions',
        'Value (\pounds M)'
    ]

    return df
