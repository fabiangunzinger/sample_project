decisions_registry = set()
features_registry = set()
variable_registry = set()
preproc_registry = set()


def decision(func):
    """Register targets."""
    decisions_registry.add(func)
    return func


def feature(func):
    """Register features."""
    features_registry.add(func)
    return func


def preproc(func):
    """Register feature for pre-processing before modelling."""
    preproc_registry.add(func.__name__)
    return func


def regvar(active=True):
    """Register all active features and targets."""
    def wrapper(func):
        if active:
            variable_registry.add(func)
        else:
            variable_registry.discard(func)
        return func
    return wrapper


def regpreproc(active=True):
    """Register feature to require preprocessing before modeling."""
    def wrapper(func):
        if active:
            preproc_registry.add(func.__name__)
        else:
            preproc_registry.discard(func.__name__)
        return func
    return wrapper
