import re
import pandas as pd

# Removal of Type 1 Clones (Exact clones with identical code apart from differences in whitespace, comments, and formatting)
def remove_duplicates(data):
    """Remove duplicate methods based on method content
        Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Java",keep = 'first')