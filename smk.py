import pandas as pd
df = pd.read_csv('data/cleaned_data.csv')


drop_df = df.copy()

drop_df = drop_df.dropna()

X_d = drop_df[['age','days_on_platform','income']]
y_d = drop_df['lifetime_value']


X_train_d = X_d[:4000]
y_train_d = y_d[:4000]

X_test_d = X_d[1000:]
y_test_d = y_d[1000:]

"""### Mean/Median/Mode Imputation

The next is mean/median/mode imputation. We can use the native numpy functions for the mean and median. We can use scipy for the mode. Then, pandas as a native `fillna` method we can use to impute the nulls with the mean/median/mode:
"""

m_df = df.copy()

X_m = m_df[['age','days_on_platform','income']]
y_m = m_df['lifetime_value']


X_train_m = X_m[:4000]
y_train_m = y_m[:4000]

X_test_m = X_m[1000:]
y_test_m = y_m[1000:]

## Mean
X_train_m.loc[:,'age'] = X_train_m['age'].fillna(np.mean(X_train_m['age']))
X_test_m.loc[:,'age'] = X_test_m['age'].fillna(np.mean(X_train_m['age'])) ## Cannot use training dataset to impute


X_train_m.loc[:,'days_on_platform'] = X_train_m['days_on_platform'].fillna(np.mean(X_train_m['days_on_platform']))
X_test_m.loc[:,'days_on_platform'] = X_test_m['days_on_platform'].fillna(np.mean(X_train_m['days_on_platform'])) ## Cannot use training dataset to impute

## Median
m_df.loc[:,'age'] = df['age'].fillna(np.median(m_df['age']))

## Mode
m_df.loc[:,'age'] = m_df['age'].fillna(stats.mode(m_df['age'])[0][0])


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

## Target - Purchases in the first six months

r_df = df.copy()

X_r = r_df[['age','days_on_platform','income']]
y_r = r_df['lifetime_value']


X_train_r = X_r[:4000]
y_train_r = y_r[:4000]

X_test_r = X_r[1000:]
y_test_r = y_r[1000:]


Imp = IterativeImputer(max_iter=10, random_state = 0)
Imp.fit(X_train_r)

X_train_r = Imp.transform(X_train_r)
X_test_r = Imp.transform(X_test_r)

X_train_r = pd.DataFrame(X_train_r)
X_train_r.columns = X_train_r.columns

X_test_r = pd.DataFrame(X_test_r)
X_test_r.columns = X_test_r.columns

r_df = pd.concat([X_train_r,X_test_r],axis = 0)
