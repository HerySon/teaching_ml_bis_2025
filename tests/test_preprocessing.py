import pandas as pd
import numpy as np
from scripts.data.preprocessing import handle_duplicates, process_categorical_columns

def test_handle_duplicates():
    # Create a sample DataFrame with duplicates
    df = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': ['a', 'b', 'b', 'c']
    })
    
    # Test analyze strategy
    df_result, info = handle_duplicates(df, strategy='analyze')
    assert df_result.equals(df)
    assert info['duplicate_count'] == 1
    
    # Test remove_all strategy
    df_result, info = handle_duplicates(df, strategy='remove_all')
    assert len(df_result) == 3
    assert info['action_taken'] == 'removed_all_duplicates'

def test_process_categorical_columns():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c'] * 2,
        'cat2': ['x', 'x', 'y'] * 2,
        'num': [1, 2, 3, 4, 5, 6]
    })
    
    # Test with default parameters
    df_result, info = process_categorical_columns(df, max_categories=5)
    assert 'cat1' in info['categorical_nominal']
    assert 'cat2' in info['categorical_nominal']
    assert len(info['dropped_columns']) == 0 