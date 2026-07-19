
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

DATA_PATH = "train_aWnotuB.csv"   
OUTPUT_DIR = "outputs"
TEST_DAYS = 30                     
LSTM_LOOKBACK = 24                 

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    return df



HOLIDAYS = pd.to_datetime([
    "2015-01-26", "2015-08-15", "2015-10-02", "2015-10-22", "2015-11-11",
    "2015-12-25", "2016-01-26", "2016-08-15", "2016-10-02", "2016-10-11",
    "2016-10-30", "2016-12-25", "2017-01-26", "2017-08-15", "2017-10-02",
])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.day
    df["month"] = df["DateTime"].dt.month
    df["year"] = df["DateTime"].dt.year
    df["weekday"] = df["DateTime"].dt.weekday          
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = df["DateTime"].dt.normalize().isin(HOLIDAYS).astype(int)
    df["is_special_day"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    return df

def run_eda(df: pd.DataFrame):
   
    fig, ax = plt.subplots(figsize=(12, 5))
    for junction, sub in df.groupby("Junction"):
        daily = sub.set_index("DateTime")["Vehicles"].resample("D").mean()
        ax.plot(daily.index, daily.values, label=f"Junction {junction}")
    ax.set_title("Average Daily Traffic per Junction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vehicles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/eda_daily_trend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    grp = df.groupby(["is_special_day", "hour"])["Vehicles"].mean().unstack(0)
    grp.plot(ax=ax)
    ax.set_title("Average Hourly Traffic: Normal Day vs. Weekend/Holiday")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Vehicles")
    ax.legend(["Normal Day", "Weekend/Holiday"])
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/eda_hourly_pattern.png", dpi=150)
    plt.close(fig)

    print(f"EDA plots saved to '{OUTPUT_DIR}/'.")


def time_based_split(sub: pd.DataFrame, test_days: int = TEST_DAYS):
    cutoff = sub["DateTime"].max() - pd.Timedelta(days=test_days)
    train = sub[sub["DateTime"] <= cutoff].copy()
    test = sub[sub["DateTime"] > cutoff].copy()
    return train, test


def run_sarima(train: pd.DataFrame, test: pd.DataFrame):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train_series = (
        train.set_index("DateTime")["Vehicles"]
        .asfreq(pd.Timedelta(hours=1))  
        .interpolate()
        .bfill()                          
        .ffill()                          
    )
    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24), 
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test))
    return forecast.values


FEATURES = ["hour", "day", "month", "weekday", "is_weekend", "is_holiday", "is_special_day"]


def run_random_forest(train: pd.DataFrame, test: pd.DataFrame):
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["Vehicles"])
    return model.predict(test[FEATURES])


def make_sequences(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def run_lstm(train: pd.DataFrame, test: pd.DataFrame, lookback: int = LSTM_LOOKBACK):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[["Vehicles"]]).flatten()
    combined = np.concatenate([train_scaled[-lookback:], scaler.transform(test[["Vehicles"]]).flatten()])

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))

    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    X_test, _ = make_sequences(combined, lookback)
    X_test = X_test.reshape((X_test.shape[0], lookback, 1))
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    return preds[: len(test)]


def evaluate(y_true, y_pred, model_name: str, junction) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"Junction": junction, "Model": model_name, "RMSE": round(rmse, 2), "MAE": round(mae, 2)}


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    df = engineer_features(df)

    print("Running EDA...")
    run_eda(df)

    results = []
    forecasts_for_plot = {}

    for junction, sub in df.groupby("Junction"):
        print(f"\n--- Junction {junction} ---")
        train, test = time_based_split(sub)
        if len(test) == 0 or len(train) < LSTM_LOOKBACK + 10:
            print("  Not enough data for this junction, skipping.")
            continue

        y_true = test["Vehicles"].values

        
        try:
            preds_sarima = run_sarima(train, test)
            results.append(evaluate(y_true, preds_sarima, "SARIMA", junction))
        except Exception as e:
            print(f"  SARIMA failed: {e}")
            preds_sarima = None

        
        try:
            preds_rf = run_random_forest(train, test)
            results.append(evaluate(y_true, preds_rf, "Random Forest", junction))
        except Exception as e:
            print(f"  Random Forest failed: {e}")
            preds_rf = None

        
        try:
            preds_lstm = run_lstm(train, test)
            results.append(evaluate(y_true[: len(preds_lstm)], preds_lstm, "LSTM", junction))
        except Exception as e:
            print(f"  LSTM failed (is tensorflow installed?): {e}")
            preds_lstm = None

        forecasts_for_plot[junction] = {
            "dates": test["DateTime"].values,
            "actual": y_true,
            "SARIMA": preds_sarima,
            "Random Forest": preds_rf,
            "LSTM": preds_lstm,
        }

    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    print("\n=== Model comparison (RMSE / MAE) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved to '{OUTPUT_DIR}/model_comparison.csv'.")

  
    for junction, data in forecasts_for_plot.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data["dates"], data["actual"], label="Actual", color="black")
        for model_name in ["SARIMA", "Random Forest", "LSTM"]:
            if data[model_name] is not None:
                ax.plot(data["dates"][: len(data[model_name])], data[model_name], label=model_name, alpha=0.8)
        ax.set_title(f"Junction {junction}: Actual vs. Forecasted Traffic")
        ax.set_xlabel("Date")
        ax.set_ylabel("Vehicles")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/forecast_junction_{junction}.png", dpi=150)
        plt.close(fig)

    print(f"Forecast plots saved to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
