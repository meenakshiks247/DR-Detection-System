import pandas as pd
import os
import sys
csv = r"D:\glucoma cataract dataset\archive\full_df.csv"
src = r"D:\glucoma cataract dataset\archive\preprocessed_images"
try:
    df = pd.read_csv(csv)
except Exception as e:
    print('ERROR reading CSV:', e)
    sys.exit(1)
print('Columns:', list(df.columns))
print('\nFirst 5 rows:')
print(df.head(5).to_string())
print('\nExistence checks for first 5 rows:')
for i, row in df.head(5).iterrows():
    L = row.get('Left-Fundus') if 'Left-Fundus' in df.columns else None
    R = row.get('Right-Fundus') if 'Right-Fundus' in df.columns else None
    print('Row', i)
    if L is not None:
        Ls = os.path.join(src, str(L))
        print(' Left:', L, '->', os.path.exists(Ls))
    else:
        print(' Left-Fundus column missing')
    if R is not None:
        Rs = os.path.join(src, str(R))
        print(' Right:', R, '->', os.path.exists(Rs))
    else:
        print(' Right-Fundus column missing')
