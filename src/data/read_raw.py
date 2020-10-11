import pandas as pd


def read(path):
    dtypes = {
        'Transaction Reference': 'int32',
        'User Reference': 'int32',
        'Year of Birth': 'float32',
        'Salary Range': 'category',
        'Postcode': 'category',
        'LSOA': 'category',
        'MSOA': 'category',
        'Derived Gender': 'category',
        'Account Reference': 'int32',
        'Provider Group Name': 'category',
        'Account Type': 'category',
        'Latest Balance': 'float32',
        'Transaction Description': 'category',
        'Credit Debit': 'category',
        'Amount': 'float32',
        'User Precedence Tag Name': 'category',
        'Manual Tag Name': 'category',
        'Auto Purpose Tag Name': 'category',
        'Merchant Name': 'category',
        'Merchant Business Line': 'category',
        'Transaction Updated Flag': 'category',
    }
    dates = [
        'User Registration Date',
        'Transaction Date',
        'Account Created Date',
        'Account Last Refreshed',
        'Data Warehouse Date Created',
        'Data Warehouse Date Last Updated',
    ]
    return pd.read_csv(path, sep='|', parse_dates=dates, dtype=dtypes)


def clean_names(df):
    df.columns = (df.columns
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('.', '_')
                  .str.strip())
    return df


def rename(df):
    new_names = {
        "user_reference": "user_id",
        "transaction_reference": "transaction_id",
        "account_reference": "account_id",
        "provider_group_name": "bank",
        "account_created_date": "account_created",
        "latest_recorded_balance": "latest_balance",
        "manual_tag_name": "manual_tag",
        "auto_purpose_tag_name": "auto_tag",
        "user_precedence_tag_name": "up_tag",
        "derived_gender": "gender",
    }
    return df.rename(columns=new_names)


def read_raw(path):
    return (
        read(path)
        .pipe(clean_names)
        .pipe(rename)
    )
