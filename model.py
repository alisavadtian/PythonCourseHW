import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\downloads\train_data\realty_data.csv')

df['postcode'] = df['postcode'].fillna(df['postcode'].median())
df['rooms'] = df['rooms'].fillna(df['rooms'].median())
df['total_square'] = df['total_square'].fillna(df['total_square'].median())

X = df[['postcode', 'rooms', 'total_square']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, r"D:\downloads\train_data\rf_model.pkl")


