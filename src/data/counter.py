from functools import wraps
from collections import Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


count = OrderedCounter()


def counter(func):
    """Count sample after each selection function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        count.update({func.__name__ + '-users': df.user_id.nunique(),
                      func.__name__ + '-acc': df.account_id.nunique(),
                      func.__name__ + '-txs': len(df),
                      func.__name__ + '-value': df.amount.sum() / 1e6})
        return df
    return wrapper


def add_count(df, step='start'):
    """Count sample at any step in pipeline."""
    count.update({step + '-users': df.user_id.nunique(),
                  step + '-acc': df.account_id.nunique(),
                  step + '-txs': len(df),
                  step + '-value': df.amount.sum() / 1e6})
    return df
