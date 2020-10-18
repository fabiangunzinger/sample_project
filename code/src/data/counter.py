from functools import wraps
from collections import Counter, OrderedDict
import re


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


count = OrderedCounter()


def counter(func):
    """Count sample after each selection function.
    Use first line of function docstring for description.
    """
    @ wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        docstr = re.match('[^\n]*', func.__doc__).group()
        count.update({
            docstr + '@users': df.user_id.nunique(),
            docstr + '@accs': df.account_id.nunique(),
            docstr + '@txns': len(df),
            docstr + '@value': df.amount.sum() / 1e6
        })
        return df
    return wrapper


def add_count(df, step):
    """Count sample at step in pipeline."""
    count.update({
        step + '@users': df.user_id.nunique(),
        step + '@accs': df.account_id.nunique(),
        step + '@txns': len(df),
        step + '@value': df.amount.sum() / 1e6
    })
    return df
