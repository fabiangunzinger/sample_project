def calculate_salaries(df):
    """
    Return monthly and yearly salary for each user.

    Salaries are monthly/yearly aggregated sums of
    all transactions tagged as salaries by MDB.
    """
    def make_groupers(df):
        """Create user-year-month/user-year strings for merge."""
        y = df.transaction_date.dt.to_period('Y').astype('str')
        ym = df.transaction_date.dt.to_period('M').astype('str')
        str_id = df.user_id.astype('str')
        df['month_grouper'] = str_id + ' - ' + ym
        df['year_grouper'] = str_id + ' - ' + y
        return df

    def calc_monthly_salary(df):
        """Return monthly salary for each user."""
        tagsum = df[['auto_tag', 'manual_tag', 'tag']].sum(1)
        mask = tagsum.str.contains('salary')
        monthly_salary = (
            df[mask]
            .set_index(['user_id', 'transaction_date'])
            .groupby(level='user_id')
            .resample('M', level='transaction_date')
            .amount.sum().abs()
            .reset_index()
            .pipe(make_groupers)
            .drop(['transaction_date', 'user_id', 'year_grouper'], axis=1)
            .rename(columns={'amount': 'monthly_salary'})
        )
        return df.merge(monthly_salary, how='left', validate='m:1')

    def calc_yearly_salary(df):
        """Return yearly salary for each user."""
        tagsum = df[['auto_tag', 'manual_tag', 'tag']].sum(1)
        mask = tagsum.str.contains('salary')
        yearly_salary = (
            df[mask]
            .set_index(['user_id', 'transaction_date'])
            .groupby(level='user_id')
            .resample('Y', level='transaction_date')
            .amount.sum().abs()
            .reset_index()
            .pipe(make_groupers)
            .drop(['transaction_date', 'user_id', 'month_grouper'], axis=1)
            .rename(columns={'amount': 'yearly_salary'})
        )
        return df.merge(yearly_salary, how='left', validate='m:1')

    def drop_groupers(df):
        return df.drop(['month_grouper', 'year_grouper'], axis=1)

    def replace_nans(df):
        df['monthly_salary'] = df.monthly_salary.fillna(0)
        df['yearly_salary'] = df.yearly_salary.fillna(0)
        return df

    return (
        df
        .pipe(make_groupers)
        .pipe(calc_monthly_salary)
        .pipe(calc_yearly_salary)
        .pipe(drop_groupers)
        .pipe(replace_nans)
    )
