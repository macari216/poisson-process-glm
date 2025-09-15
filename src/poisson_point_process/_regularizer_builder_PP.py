"""Utility functions for creating regularizer object."""

AVAILABLE_REGULARIZERS = ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]


def create_regularizer(name: str):
    """
    Create a regularizer from a given name.

    Parameters
    ----------
    name :
        The string name of the regularizer to create. Must be one of: 'UnRegularized', 'Ridge', 'Lasso', 'GroupLasso'.

    Returns
    -------
    :
        The regularizer instance with default parameters.

    Raises
    ------
    ValueError
        If the `name` provided does not match to any available regularizer.
    """
    if name == "UnRegularized":
        from .regularizer_PP import UnRegularized

        return UnRegularized()
    elif name == "Ridge":
        from .regularizer_PP import Ridge

        return Ridge()
    elif name == "Lasso":
        from .regularizer_PP import Lasso

        return Lasso()
    elif name == "GroupLasso":
        from .regularizer_PP import GroupLasso

        return GroupLasso()
    else:
        raise ValueError(
            f"Unknown regularizer: {name}. "
            f"Regularizer must be one of {AVAILABLE_REGULARIZERS}"
        )
