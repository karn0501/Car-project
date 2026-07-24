"""
Phase 7 Training Script: LSTM Price Trend Forecaster.
Generates synthetic price history from existing listings, trains 2-layer LSTM
with macroeconomic signals, and saves model checkpoint.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.data_exporter import load_dataset_from_db
from src.trend_forecaster import PriceTrendForecaster


def train_trend_model():
    print("=" * 80)
    print("PHASE 7: LSTM PRICE TREND FORECASTER TRAINING")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load listing data
    print("\n[Step 1/4] Loading listings from database...")
    df = load_dataset_from_db()
    required_cols = ["company_name", "model_name", "asking_price", "city"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  \\-- ERROR: Missing columns: {missing}")
        return

    print(f"  \\-- Loaded {len(df)} listings across {df['company_name'].nunique()} brands")

    # 2. Generate synthetic price history
    print("\n[Step 2/4] Generating synthetic monthly price history (24 months)...")
    forecaster = PriceTrendForecaster(model_path="models/price_trend_lstm.pt")
    history_df = forecaster.generate_price_history(df, months=24)
    unique_variants = history_df["variant_key"].nunique()
    print(f"  |-- Generated {len(history_df)} monthly records")
    print(f"  \\-- Covering {unique_variants} unique variant-city combinations")

    # 3. Train LSTM model
    print("\n[Step 3/4] Training 2-Layer LSTM model (30 epochs)...")
    metrics = forecaster.train(history_df, epochs=30, lr=0.001)

    if metrics["status"] == "success":
        print(f"  |-- Training Sequences : {metrics['num_sequences']}")
        print(f"  |-- Final Train Loss   : {metrics['final_train_loss']:.6f}")
        print(f"  |-- Best Val Loss      : {metrics['best_val_loss']:.6f}")
        print(f"  \\-- Epochs Completed   : {metrics['epochs']}")
    else:
        print(f"  \\-- Training Status: {metrics['status']}")

    # 4. Test forecast on sample variant
    print("\n[Step 4/4] Running sample 3-month forecast...")
    sample_variant = history_df["variant_key"].iloc[0] if len(history_df) > 0 else None
    if sample_variant:
        sample_data = history_df[history_df["variant_key"] == sample_variant].tail(12)
        forecast_result = forecaster.forecast(sample_data, months_ahead=3)

        if forecast_result["status"] == "success":
            print(f"  |-- Variant: {sample_variant}")
            print(f"  |-- Base Price: INR {forecast_result['base_price']:,.0f}")
            print(f"  |-- Trend Direction: {forecast_result['trend_direction']}")
            for fc in forecast_result["forecasts"]:
                direction = "↑" if fc["change_pct"] > 0 else ("↓" if fc["change_pct"] < 0 else "→")
                print(f"      Month {fc['month']}: INR {fc['predicted_price']:,.0f} ({direction} {fc['change_pct']:+.2f}%)")
        else:
            print(f"  \\-- Forecast: {forecast_result['status']}")
    else:
        print("  \\-- No variant data available for sample forecast")

    # Summary
    print("\n" + "=" * 80)
    print("LSTM TREND FORECASTER TRAINING COMPLETE")
    print("=" * 80)
    print(f"  |-- Model Architecture : 2-Layer LSTM (hidden=64, input=6)")
    print(f"  |-- Macro Signals      : Fuel Index, Interest Rate, Festive Season")
    print(f"  |-- Sequence Length     : 6 months sliding window")
    print(f"  |-- Saved Checkpoint   : models/price_trend_lstm.pt")
    print("=" * 80)


if __name__ == "__main__":
    train_trend_model()
