"""
Phase 7 Module: LSTM/GRU Price Trend Forecaster.
Predicts future used car price trajectories using time-series data from the
price_history table, enriched with macroeconomic signals (fuel prices,
interest rates, festive season markers).

Uses a 2-layer LSTM with sliding window sequences for robust forecasting.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn


# ─── Macroeconomic Signal Generators ───────────────────────────────────────────

def get_fuel_price_index(date: datetime) -> float:
    """
    Synthetic fuel price index normalized to [0, 1].
    In production, this would pull from a real API (e.g., PPAC, petroleum.nic.in).
    Models seasonal fuel price fluctuations.
    """
    day_of_year = date.timetuple().tm_yday
    # Simulate seasonal fuel price pattern (higher in summer, lower in winter)
    base = 0.5
    seasonal = 0.15 * np.sin(2 * np.pi * day_of_year / 365)
    # Gradual upward trend over years
    year_factor = 0.02 * (date.year - 2020)
    return round(max(0.0, min(1.0, base + seasonal + year_factor)), 4)


def get_interest_rate_index(date: datetime) -> float:
    """
    Synthetic interest rate index normalized to [0, 1].
    Higher rates → lower used car demand → lower prices.
    In production, this would pull from RBI data.
    """
    month = date.month
    # Simulate gradual rate changes
    base = 0.45
    cycle = 0.1 * np.sin(2 * np.pi * month / 12 + np.pi / 4)
    return round(max(0.0, min(1.0, base + cycle)), 4)


def get_festive_season_marker(date: datetime) -> float:
    """
    Returns 1.0 during festive months (Oct-Nov for Diwali/Dussehra,
    March-April for Holi/New Year), 0.0 otherwise.
    Festive seasons typically see higher demand and prices.
    """
    festive_months = {3, 4, 10, 11}
    return 1.0 if date.month in festive_months else 0.0


# ─── LSTM Model Architecture ──────────────────────────────────────────────────

class PriceTrendLSTM(nn.Module):
    """
    2-Layer LSTM network for time-series price trend forecasting.

    Input features per timestep:
        - Normalized average price
        - Fuel price index
        - Interest rate index
        - Festive season marker
        - Month sin/cos encoding (cyclical)
    """

    def __init__(self, input_size: int = 6, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Tensor of shape (batch_size, output_size) — predicted next-step price
        """
        lstm_out, _ = self.lstm(x)
        # Take the output from the last timestep
        last_hidden = lstm_out[:, -1, :]
        prediction = self.fc(last_hidden)
        return prediction


# ─── Price Trend Forecaster ────────────────────────────────────────────────────

class PriceTrendForecaster:
    """
    End-to-end price trend forecasting engine.

    Generates synthetic price history from existing listings data,
    builds LSTM-ready sequences with macroeconomic features,
    trains, and forecasts future price directions.
    """

    def __init__(self, model_path: str = "models/price_trend_lstm.pt",
                 seq_length: int = 6, hidden_size: int = 64):
        self.model_path = model_path
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.input_size = 6  # price + fuel + interest + festive + month_sin + month_cos
        self.model = PriceTrendLSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=1
        )
        self.price_scaler = {"min": 0, "max": 1}  # Will be fitted during training

        if os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load trained LSTM model weights."""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.price_scaler = checkpoint.get("price_scaler", self.price_scaler)
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load trend model from {path}: {e}")

    def generate_price_history(self, df: pd.DataFrame, months: int = 24) -> pd.DataFrame:
        """
        Generate synthetic monthly price history from existing listings data.
        Applies realistic depreciation curves and seasonal patterns.

        Args:
            df: DataFrame with columns [company_name, model_name, manufacture_year, asking_price, city]
            months: Number of months of history to generate

        Returns:
            DataFrame with columns [variant_key, city, date, avg_price, fuel_index,
                                     interest_index, festive_marker, month_sin, month_cos]
        """
        records = []
        today = datetime.now()

        # Group by model + city to create per-segment time series
        group_cols = [c for c in ["company_name", "model_name", "city"] if c in df.columns]
        if not group_cols:
            return pd.DataFrame()

        grouped = df.groupby(group_cols)

        for group_key, group_df in grouped:
            if len(group_df) < 3:
                continue

            base_price = group_df["asking_price"].median()
            variant_key = " ".join(str(k) for k in group_key) if isinstance(group_key, tuple) else str(group_key)
            city = group_key[-1] if isinstance(group_key, tuple) else "Unknown"

            for m in range(months, 0, -1):
                date = today - timedelta(days=m * 30)

                # Depreciation: ~1-2% per month with noise
                depreciation = 1.0 - (0.012 * m)
                noise = np.random.normal(0, 0.02)
                seasonal = 0.03 * np.sin(2 * np.pi * date.month / 12)

                avg_price = max(base_price * 0.3, base_price * (depreciation + noise + seasonal))

                month_rad = 2 * np.pi * date.month / 12
                records.append({
                    "variant_key": variant_key,
                    "city": city,
                    "date": date.strftime("%Y-%m-%d"),
                    "avg_price": round(avg_price, 2),
                    "fuel_index": get_fuel_price_index(date),
                    "interest_index": get_interest_rate_index(date),
                    "festive_marker": get_festive_season_marker(date),
                    "month_sin": round(np.sin(month_rad), 4),
                    "month_cos": round(np.cos(month_rad), 4),
                })

        return pd.DataFrame(records)

    def _normalize_prices(self, prices: np.ndarray, fit: bool = False) -> np.ndarray:
        """Min-max normalize prices to [0, 1]."""
        if fit:
            self.price_scaler["min"] = float(prices.min())
            self.price_scaler["max"] = float(prices.max())
        pmin = self.price_scaler["min"]
        pmax = self.price_scaler["max"]
        if pmax == pmin:
            return np.zeros_like(prices)
        return (prices - pmin) / (pmax - pmin)

    def _denormalize_prices(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized prices back to original scale."""
        pmin = self.price_scaler["min"]
        pmax = self.price_scaler["max"]
        return normalized * (pmax - pmin) + pmin

    def build_sequences(self, history_df: pd.DataFrame) -> tuple:
        """
        Build sliding window sequences for LSTM training.

        Returns:
            X: np.ndarray of shape (num_sequences, seq_length, input_size)
            y: np.ndarray of shape (num_sequences, 1)
        """
        all_X, all_y = [], []

        for variant_key, group in history_df.groupby("variant_key"):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < self.seq_length + 1:
                continue

            prices = group["avg_price"].values
            norm_prices = self._normalize_prices(prices, fit=True)

            features = np.column_stack([
                norm_prices,
                group["fuel_index"].values,
                group["interest_index"].values,
                group["festive_marker"].values,
                group["month_sin"].values,
                group["month_cos"].values,
            ])

            for i in range(len(features) - self.seq_length):
                X_seq = features[i:i + self.seq_length]
                y_val = norm_prices[i + self.seq_length]
                all_X.append(X_seq)
                all_y.append([y_val])

        if not all_X:
            return np.array([]).reshape(0, self.seq_length, self.input_size), np.array([]).reshape(0, 1)

        return np.array(all_X), np.array(all_y)

    def train(self, history_df: pd.DataFrame, epochs: int = 30, lr: float = 0.001) -> dict:
        """
        Train the LSTM model on price history data.

        Args:
            history_df: DataFrame from generate_price_history()
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Dict with training metrics
        """
        X, y = self.build_sequences(history_df)
        if len(X) == 0:
            return {"status": "no_data", "loss": float("inf")}

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Train/val split (80/20)
        split_idx = max(1, int(len(X_tensor) * 0.8))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.model.train()

        best_val_loss = float("inf")
        train_losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val) if len(X_val) > 0 else torch.tensor([0.0])
                val_loss = criterion(val_preds, y_val).item() if len(X_val) > 0 else 0.0
            self.model.train()

            train_losses.append(loss.item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Save model
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else "models", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "price_scaler": self.price_scaler,
        }, self.model_path)

        self.model.eval()
        return {
            "status": "success",
            "final_train_loss": round(train_losses[-1], 6),
            "best_val_loss": round(best_val_loss, 6),
            "num_sequences": len(X),
            "epochs": epochs,
        }

    def forecast(self, recent_history: pd.DataFrame, months_ahead: int = 3) -> dict:
        """
        Forecast future price trend for a specific variant/city.

        Args:
            recent_history: DataFrame with most recent seq_length months of data
            months_ahead: Number of months to forecast

        Returns:
            Dict with forecasted prices and trend direction
        """
        self.model.eval()

        if len(recent_history) < self.seq_length:
            return {"status": "insufficient_data", "forecasts": []}

        # Build feature sequence from recent history
        recent = recent_history.sort_values("date").tail(self.seq_length)
        prices = recent["avg_price"].values
        norm_prices = self._normalize_prices(prices)

        features = np.column_stack([
            norm_prices,
            recent["fuel_index"].values,
            recent["interest_index"].values,
            recent["festive_marker"].values,
            recent["month_sin"].values,
            recent["month_cos"].values,
        ])

        forecasts = []
        current_seq = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, input_size)
        last_price = prices[-1]

        for m in range(months_ahead):
            with torch.no_grad():
                pred_norm = self.model(current_seq).item()

            pred_price = self._denormalize_prices(np.array([pred_norm]))[0]
            pred_price = max(pred_price, last_price * 0.5)  # Floor at 50% of last price

            future_date = datetime.now() + timedelta(days=(m + 1) * 30)
            month_rad = 2 * np.pi * future_date.month / 12

            forecasts.append({
                "month": m + 1,
                "date": future_date.strftime("%Y-%m"),
                "predicted_price": round(pred_price, 0),
                "change_pct": round(((pred_price - last_price) / last_price) * 100, 2),
            })

            # Build next timestep features and slide window
            new_step = torch.FloatTensor([[
                pred_norm,
                get_fuel_price_index(future_date),
                get_interest_rate_index(future_date),
                get_festive_season_marker(future_date),
                round(np.sin(month_rad), 4),
                round(np.cos(month_rad), 4),
            ]])
            current_seq = torch.cat([current_seq[:, 1:, :], new_step.unsqueeze(0)], dim=1)

        # Determine overall trend
        if forecasts:
            final_change = forecasts[-1]["change_pct"]
            if final_change > 2:
                trend = "RISING"
            elif final_change < -2:
                trend = "FALLING"
            else:
                trend = "STABLE"
        else:
            trend = "UNKNOWN"

        return {
            "status": "success",
            "trend_direction": trend,
            "forecasts": forecasts,
            "base_price": round(last_price, 0),
        }
