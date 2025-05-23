import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

st.title("NYC Uber Fare Predictor 🚕🗽")
@st.cache_data 

def load():
    df = pd.read_csv('uber.csv', nrows=50000)
    df = df.dropna()
    df = df[(df['fare_amount'] > 0)]
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    return df

df = load()


def process(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour

    def calculate_distance(row):
        return geodesic(
            (row['pickup_latitude'], row['pickup_longitude']),
            (row['dropoff_latitude'], row['dropoff_longitude'])
            ).km

    df['distance'] = df.apply(calculate_distance, axis=1)
    return df

df = process(df)

#log transform the target for better performance
target = np.log1p(df['fare_amount'])  # log1p to avoid log(0)

features = df[['distance', 'hour']] 

#train test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#training random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#prediction in log scale
y_pred_log = model.predict(X_test)

#use inverse log to get actual fare predictions
y_pred = np.expm1(y_pred_log)
y_test_exp = np.expm1(y_test)

#user inputs
st.sidebar.subheader("Select the following to predict your Uber Fare: ")
distance = st.sidebar.slider("Distance (km)", 0.0, 20.0, 5.0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

input_data = pd.DataFrame([[distance, hour]], columns=['distance', 'hour'])
predicted_fare_log = model.predict(input_data)[0]
predicted_fare = np.expm1(predicted_fare_log)
st.metric("Estimated Fare (USD)", f"${predicted_fare:.2f}")

#filter outliers (fares over $100)
filtered = pd.DataFrame({'Actual': y_test_exp, 'Predicted': y_pred})
filtered = filtered[(filtered['Actual'] <= 100) & (filtered['Predicted'] <= 100)]

#some metrics
mae = mean_absolute_error(filtered['Actual'], filtered['Predicted'])
r2 = r2_score(filtered['Actual'], filtered['Predicted'])

st.subheader("Model Performance")
st.write(f"Mean Absolute Error (MAE): ${mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

#plot graph
st.subheader('Actual vs Predicted Fare')
fig, ax = plt.subplots()
ax.scatter(filtered['Actual'], filtered['Predicted'], alpha=0.5)
ax.plot([0, 100], [0, 100], 'r--')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("Actual Fare")
ax.set_ylabel("Predicted Fare")
ax.set_title("Actual vs Predicted Fares")
st.pyplot(fig)

#create log fare column
if 'log_fare_amount' in df:
    df = df.drop('log_fare_amount', axis=1)
df['log_fare_amount'] = np.log1p(df['fare_amount'])

st.subheader('Fare Distribution on Weekdays ')
weekdays = df[df['pickup_datetime'].dt.dayofweek < 5]

#create bar graph 
plt.figure(figsize=(10, 6))
sns.histplot(weekdays['fare_amount'], bins = 99, kde=False, color='orange')
plt.title('Distribution of Fares (Weekdays)')
plt.xlabel('Fare Amount ($ USD)')
plt.ylabel('Frequency')
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, 10))
st.pyplot(plt) 

link = "https://www.kaggle.com/datasets/yasserh/uber-fares-dataset"
st.write("➡️ View the dataset I used [here](%s)" % link)