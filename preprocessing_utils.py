
import numpy as np

"""
Utility transformer functions for the house price pipeline.
Must live in a proper module (not __main__) so joblib/pickle
can serialize references to these functions reliably.
"""

def log1p_dataframe(X):
    """Applies log1p elementwise, preserving DataFrame structure."""
    return np.log1p(X)
