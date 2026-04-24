"""Configuration settings for FinRL-Trading.

This module centralizes all configuration parameters including
data sources, trading environment settings, and model hyperparameters.
Values are loaded from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Data source configuration
# ---------------------------------------------------------------------------
DATA_SOURCE = os.getenv("DATA_SOURCE", "yahoofinance")

# Date ranges
TRAIN_START_DATE = os.getenv("TRAIN_START_DATE", "2015-01-01")
TRAIN_END_DATE = os.getenv("TRAIN_END_DATE", "2020-12-31")
VALID_START_DATE = os.getenv("VALID_START_DATE", "2021-01-01")
VALID_END_DATE = os.getenv("VALID_END_DATE", "2021-12-31")
TEST_START_DATE = os.getenv("TEST_START_DATE", "2022-01-01")
TEST_END_DATE = os.getenv("TEST_END_DATE", "2023-01-01")

# Default ticker universe (Dow Jones 30 subset for quick testing)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "TSLA", "NVDA", "JPM", "JNJ", "V",
]

# ---------------------------------------------------------------------------
# Trading environment
# ---------------------------------------------------------------------------
INITIAL_AMOUNT = float(os.getenv("INITIAL_AMOUNT", "1_000_000"))
TRANSACTION_COST_PCT = float(os.getenv("TRANSACTION_COST_PCT", "0.001"))  # 0.1 %
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0005"))                # 0.05 %
MAX_STOCK_QUANTITY = int(os.getenv("MAX_STOCK_QUANTITY", "100"))
REWARD_SCALING = float(os.getenv("REWARD_SCALING", "1e-4"))

# Technical indicators computed for each ticker
TECH_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# ---------------------------------------------------------------------------
# RL model hyperparameters
# ---------------------------------------------------------------------------
RL_AGENT = os.getenv("RL_AGENT", "ppo")  # ppo | a2c | ddpg | td3 | sac

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 2.5e-4,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "verbose": 0,
}

A2C_PARAMS = {
    "n_steps": 5,
    "ent_coef": 0.005,
    "learning_rate": 7e-4,
    "verbose": 0,
}

DDPG_PARAMS = {
    "buffer_size": 100_000,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "verbose": 0,
}

TD3_PARAMS = {
    "buffer_size": 100_000,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "verbose": 0,
}

# NOTE: Using a fixed ent_coef of 0.1 instead of "auto_0.1" to keep entropy
# tuning disabled while I experiment with manual coefficient values.
SAC_PARAMS = {
    "buffer_size": 100_000,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "ent_coef": 0.1,
    "verbose": 0,
}

AGENT_PARAMS: dict = {
    "ppo": PPO_PARAMS,
    "a2c": A2C_PARAMS,
    "ddpg": DDPG_PARAMS,
    "td3": TD3_PARAMS,
    "sac": SAC_PARAMS,
}

# Total training timesteps
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500_000"))
