import pandas as pd

sample_df = pd.DataFrame({'one': pd.Series([1., 2., 3., 5.]),
                          'two': pd.Series([1., 2., 3., 4.]),
                          'three': pd.Series(['cat1', 'cat1', 'cat2', 'cat3'], dtype="category")})
