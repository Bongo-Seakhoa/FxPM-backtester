"""
FX Portfolio Manager - Core Components
======================================

Core classes and utilities for the FX Portfolio Production Pipeline.
Contains: Configuration, Instrument Specifications, Data Loading, Feature Engineering,
Data Splitting, and Backtesting Engine.

Version: 2.1 (Corrected)
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import time

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Global pipeline configuration."""
    data_dir: Path = Path("./sample_data")
    output_dir: Path = Path("./fx_portfolio_outputs")
    
    # Data split ratios
    train_pct: float = 70.0
    val_pct: float = 15.0
    test_pct: float = 15.0
    
    # Backtest settings
    initial_capital: float = 10000.0
    risk_per_trade_pct: float = 1.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.5  # Additional slippage beyond spread
    
    # Optimization settings
    max_param_combos: int = 50
    min_trades: int = 20
    min_robustness: float = 0.20
    
    # Timeframes to test
    timeframes: List[str] = field(default_factory=lambda: ['M30', 'H1', 'H4', 'D1'])


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class Timer:
    """Context manager for timing operations."""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"⏱️  {self.name}: {elapsed:.2f}s")


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class StrategyCategory(Enum):
    """Strategy categories."""
    TREND_FOLLOWING = "trend"
    MEAN_REVERSION = "mr"
    BREAKOUT_MOMENTUM = "breakout"
    VOLATILITY = "volatility"
    HYBRID = "hybrid"


class TradeStatus(Enum):
    """Trade status types."""
    OPEN = "open"
    CLOSED_TP = "closed_tp"
    CLOSED_SL = "closed_sl"
    CLOSED_SIGNAL = "closed_signal"
    CLOSED_TIME = "closed_time"
    CLOSED_MANUAL = "closed_manual"


# ============================================================================
# INSTRUMENT SPECIFICATIONS (CORRECTED)
# ============================================================================

@dataclass
class InstrumentSpec:
    """
    Specification for a trading instrument.
    
    IMPORTANT: pip_value represents USD per pip per STANDARD lot.
    
    For FX pairs: 1 standard lot = 100,000 units of base currency
    For Gold (XAUUSD): 1 lot = 100 troy oz
    For Silver (XAGUSD): 1 lot = 5,000 oz
    """
    symbol: str
    pip_position: int  # Decimal position (4 for EURUSD, 2 for USDJPY/XAUUSD)
    pip_value: float = 10.0  # USD per pip per standard lot
    spread_avg: float = 1.0  # Average spread in pips
    min_lot: float = 0.01
    max_lot: float = 100.0
    commission_per_lot: float = 7.0  # Commission in USD per round trip per lot
    swap_long: float = 0.0  # Daily swap for long positions (in account currency)
    swap_short: float = 0.0  # Daily swap for short positions
    
    @property
    def pip_size(self) -> float:
        """Price change for one pip."""
        return 10 ** (-self.pip_position)
    
    def price_to_pips(self, price_diff: float) -> float:
        """Convert price difference to pips."""
        return price_diff / self.pip_size
    
    def pips_to_price(self, pips: float) -> float:
        """Convert pips to price difference."""
        return pips * self.pip_size
    
    def get_half_spread_price(self) -> float:
        """Get half spread as price (for bid/ask calculations)."""
        return self.pips_to_price(self.spread_avg / 2.0)
    
    def get_entry_price(self, mid_price: float, is_long: bool) -> float:
        """
        Calculate entry price including spread.
        - Longs enter at ASK (mid + half_spread)
        - Shorts enter at BID (mid - half_spread)
        """
        half_spread = self.get_half_spread_price()
        return mid_price + half_spread if is_long else mid_price - half_spread
    
    def get_exit_price(self, mid_price: float, is_long: bool) -> float:
        """
        Calculate exit price including spread.
        - Longs exit at BID (mid - half_spread)
        - Shorts exit at ASK (mid + half_spread)
        """
        half_spread = self.get_half_spread_price()
        return mid_price - half_spread if is_long else mid_price + half_spread


# Default instrument specifications
# These are used when MT5 data is not available (e.g., backtesting from CSV)
# Live trading will override these with actual broker values
INSTRUMENT_SPECS = {
    # Major FX Pairs
    'EURUSD': InstrumentSpec('EURUSD', 4, 10.0, 1.0, 0.01, 100.0, 7.0, -6.5, 1.2),
    'GBPUSD': InstrumentSpec('GBPUSD', 4, 10.0, 1.2, 0.01, 100.0, 7.0, -5.0, 0.8),
    'AUDUSD': InstrumentSpec('AUDUSD', 4, 10.0, 1.2, 0.01, 100.0, 7.0, -4.0, 0.5),
    'NZDUSD': InstrumentSpec('NZDUSD', 4, 10.0, 1.5, 0.01, 100.0, 7.0, -3.5, 0.3),
    'USDJPY': InstrumentSpec('USDJPY', 2, 9.0, 1.0, 0.01, 100.0, 7.0, 8.5, -15.0),
    'USDCAD': InstrumentSpec('USDCAD', 4, 7.5, 1.5, 0.01, 100.0, 7.0, 2.5, -8.0),
    'USDCHF': InstrumentSpec('USDCHF', 4, 11.0, 1.5, 0.01, 100.0, 7.0, 5.0, -10.0),
    
    # Cross pairs
    'AUDNZD': InstrumentSpec('AUDNZD', 4, 6.0, 2.5, 0.01, 100.0, 8.0, -2.0, -1.5),
    'EURGBP': InstrumentSpec('EURGBP', 4, 12.5, 1.5, 0.01, 100.0, 7.0, -4.0, 0.5),
    'EURJPY': InstrumentSpec('EURJPY', 2, 9.0, 1.5, 0.01, 100.0, 7.0, 3.0, -10.0),
    'GBPJPY': InstrumentSpec('GBPJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, 5.0, -12.0),
    
    # Exotic FX Pairs
    'USDBRL': InstrumentSpec('USDBRL', 4, 2.0, 50.0, 0.01, 100.0, 15.0, 15.0, -40.0),
    'USDMXN': InstrumentSpec('USDMXN', 4, 0.55, 30.0, 0.01, 100.0, 12.0, 20.0, -45.0),
    'USDPLN': InstrumentSpec('USDPLN', 4, 2.5, 25.0, 0.01, 100.0, 10.0, 3.0, -12.0),
    'USDNOK': InstrumentSpec('USDNOK', 4, 0.95, 30.0, 0.01, 100.0, 10.0, 2.0, -10.0),
    'USDSGD': InstrumentSpec('USDSGD', 4, 7.5, 2.0, 0.01, 100.0, 8.0, 1.5, -6.0),
    'USDTRY': InstrumentSpec('USDTRY', 4, 0.035, 100.0, 0.01, 100.0, 20.0, 50.0, -150.0),
    'USDZAR': InstrumentSpec('USDZAR', 4, 0.55, 50.0, 0.01, 100.0, 15.0, 12.0, -35.0),
    
    # Commodities
    # XAUUSD: 1 lot = 100 troy oz, pip_size = 0.01, pip_value = $1.00
    'XAUUSD': InstrumentSpec('XAUUSD', 2, 1.0, 3.0, 0.01, 100.0, 5.0, -8.0, -5.0),
    # XAGUSD: 1 lot = 5,000 oz, pip_size = 0.001, pip_value = $5.00
    'XAGUSD': InstrumentSpec('XAGUSD', 3, 5.0, 2.0, 0.01, 100.0, 5.0, -3.0, -2.0),



# Additional FX Pairs (aligned with config)
'AUDJPY': InstrumentSpec('AUDJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, -4.0, 1.0),
'EURAUD': InstrumentSpec('EURAUD', 4, 7.0, 2.0, 0.01, 100.0, 7.0, -4.0, 0.5),
'EURCHF': InstrumentSpec('EURCHF', 4, 11.0, 2.0, 0.01, 100.0, 7.0, -3.0, 0.5),
'EURCAD': InstrumentSpec('EURCAD', 4, 7.5, 2.0, 0.01, 100.0, 7.0, -4.0, 0.5),
'EURNZD': InstrumentSpec('EURNZD', 4, 6.0, 3.0, 0.01, 100.0, 7.0, -4.0, 0.5),
'GBPAUD': InstrumentSpec('GBPAUD', 4, 7.0, 2.5, 0.01, 100.0, 7.0, -5.0, 0.8),
'GBPCAD': InstrumentSpec('GBPCAD', 4, 7.5, 2.5, 0.01, 100.0, 7.0, -5.0, 0.8),
'GBPCHF': InstrumentSpec('GBPCHF', 4, 11.0, 2.5, 0.01, 100.0, 7.0, -5.0, 0.8),
'CADJPY': InstrumentSpec('CADJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, -2.0, 0.5),
'NZDJPY': InstrumentSpec('NZDJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, -2.0, 0.5),

# Crypto (CFDs) - defaults used mainly for offline backtests; live trading overrides via broker specs
'ETHUSD': InstrumentSpec('ETHUSD', 2, 1.0, 8.0, 0.01, 100.0, 0.0, 0.0, 0.0),
'XRPUSD': InstrumentSpec('XRPUSD', 4, 1.0, 12.0, 0.01, 100.0, 0.0, 0.0, 0.0),
'TONUSD': InstrumentSpec('TONUSD', 3, 1.0, 12.0, 0.01, 100.0, 0.0, 0.0, 0.0),

# Indices - point-based instruments; defaults used mainly for offline backtests; live trading overrides via broker specs
'US100': InstrumentSpec('US100', 0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
'US30':  InstrumentSpec('US30',  0, 1.0, 3.0, 0.01, 100.0, 0.0, 0.0, 0.0),
'DE30':  InstrumentSpec('DE30',  0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
'EU50':  InstrumentSpec('EU50',  0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
'UK100': InstrumentSpec('UK100', 0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
'JP225': InstrumentSpec('JP225', 0, 1.0, 8.0, 0.01, 100.0, 0.0, 0.0, 0.0),
}



# ============================================================================
# DATA LOADING AND RESAMPLING
# ============================================================================

class DataLoader:
    """Loads and resamples FX data from CSV files."""
    
    RESAMPLE_MAP = {
        'M5': '5min',
        'M15': '15min',
        'M30': '30min',
        'H1': '1h',
        'H4': '4h',
        'D1': '1D',
    }
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.resampled_cache: Dict[str, pd.DataFrame] = {}
    
    def load_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from data directory."""
        print(f"Loading data from: {self.data_dir}")
        
        for filepath in self.data_dir.glob("*.csv"):
            symbol = filepath.stem.split('_')[0].upper()
            
            try:
                df = pd.read_csv(filepath)
                
                # Standardize column names
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Parse datetime
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                
                # Standardize OHLC column names
                # First pass: identify columns
                rename_map = {}
                has_tick_volume = False
                has_volume = False
                
                for col in df.columns:
                    col_lower = col.lower()
                    if 'open' in col_lower and 'Open' not in rename_map.values():
                        rename_map[col] = 'Open'
                    elif 'high' in col_lower and 'High' not in rename_map.values():
                        rename_map[col] = 'High'
                    elif 'low' in col_lower and 'Low' not in rename_map.values():
                        rename_map[col] = 'Low'
                    elif 'close' in col_lower and 'Close' not in rename_map.values():
                        rename_map[col] = 'Close'
                    elif col_lower == 'tick_volume':
                        has_tick_volume = True
                        rename_map[col] = 'Volume'  # Prefer tick_volume
                    elif col_lower == 'volume' and not has_tick_volume:
                        has_volume = True
                        # Only use volume if tick_volume doesn't exist
                        if 'Volume' not in rename_map.values():
                            rename_map[col] = 'Volume'
                    elif 'spread' in col_lower and 'Spread' not in rename_map.values():
                        rename_map[col] = 'Spread'
                
                df.rename(columns=rename_map, inplace=True)
                
                # If we ended up with duplicate Volume columns, keep only the first (tick_volume)
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()]
                
                # Ensure required columns exist
                required = ['Open', 'High', 'Low', 'Close']
                if not all(c in df.columns for c in required):
                    print(f"  ⚠️  {symbol}: Missing OHLC columns, skipping")
                    continue
                
                # Sort by index
                df.sort_index(inplace=True)
                
                # Remove any duplicate indices
                df = df[~df.index.duplicated(keep='first')]
                
                self.raw_data[symbol] = df
                print(f"  ✅ {symbol}: {len(df):,} bars loaded")
                
            except Exception as e:
                print(f"  ❌ {symbol}: Error loading - {e}")
        
        return self.raw_data
    
    def resample(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Resample data to specified timeframe."""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.resampled_cache:
            return self.resampled_cache[cache_key]
        
        if symbol not in self.raw_data:
            return None
        
        df = self.raw_data[symbol]
        
        if timeframe not in self.RESAMPLE_MAP:
            return None
        
        rule = self.RESAMPLE_MAP[timeframe]
        
        # Resample OHLC data
        ohlc_dict = {
            'Open': df['Open'].resample(rule).first(),
            'High': df['High'].resample(rule).max(),
            'Low': df['Low'].resample(rule).min(),
            'Close': df['Close'].resample(rule).last(),
        }
        
        # Handle Volume if it exists and is a single column
        if 'Volume' in df.columns:
            vol_col = df['Volume']
            if isinstance(vol_col, pd.Series):
                ohlc_dict['Volume'] = vol_col.resample(rule).sum()
        
        ohlc_df = pd.DataFrame(ohlc_dict)
        resampled = ohlc_df.dropna()
        
        self.resampled_cache[cache_key] = resampled
        return resampled


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureComputer:
    """Computes technical indicators and features."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.feature_cache: Dict[str, pd.DataFrame] = {}
    
    def compute_all_features(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Compute all features for a symbol-timeframe pair."""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        df = self.data_loader.resample(symbol, timeframe)
        if df is None or len(df) < 250:
            return None
        
        features = df.copy()
        
        # ================================================================
        # PRICE FEATURES
        # ================================================================
        features['Returns'] = features['Close'].pct_change()
        features['Log_Returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['Range'] = features['High'] - features['Low']
        features['Body'] = abs(features['Close'] - features['Open'])
        features['Upper_Shadow'] = features['High'] - features[['Open', 'Close']].max(axis=1)
        features['Lower_Shadow'] = features[['Open', 'Close']].min(axis=1) - features['Low']
        
        # ================================================================
        # MOVING AVERAGES
        # ================================================================
        for period in [5, 8, 10, 13, 20, 21, 50, 100, 200]:
            features[f'SMA_{period}'] = features['Close'].rolling(period).mean()
            features[f'EMA_{period}'] = features['Close'].ewm(span=period, adjust=False).mean()
        
        # Hull Moving Average (period 20)
        period = 20
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = features['Close'].rolling(half_period).apply(
            lambda x: np.sum(np.arange(1, half_period + 1) * x) / np.sum(np.arange(1, half_period + 1)), raw=True)
        wma_full = features['Close'].rolling(period).apply(
            lambda x: np.sum(np.arange(1, period + 1) * x) / np.sum(np.arange(1, period + 1)), raw=True)
        raw_hma = 2 * wma_half - wma_full
        features['HMA_20'] = raw_hma.rolling(sqrt_period).apply(
            lambda x: np.sum(np.arange(1, sqrt_period + 1) * x) / np.sum(np.arange(1, sqrt_period + 1)), raw=True)
        
        # ================================================================
        # ATR (Average True Range)
        # ================================================================
        for period in [7, 10, 14, 21]:
            high_low = features['High'] - features['Low']
            high_close = abs(features['High'] - features['Close'].shift(1))
            low_close = abs(features['Low'] - features['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'ATR_{period}'] = tr.rolling(period).mean()
            features[f'ATR_{period}_pct'] = features[f'ATR_{period}'] / features['Close'] * 100
        
        # ================================================================
        # RSI (Relative Strength Index)
        # ================================================================
        for period in [7, 14, 21]:
            delta = features['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.inf)
            features[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        rsi = features['RSI_14']
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        features['StochRSI'] = (rsi - rsi_min) / (rsi_max - rsi_min)
        features['StochRSI_K'] = features['StochRSI'].rolling(3).mean()
        features['StochRSI_D'] = features['StochRSI_K'].rolling(3).mean()
        
        # ================================================================
        # MACD (Moving Average Convergence Divergence)
        # ================================================================
        ema12 = features['Close'].ewm(span=12, adjust=False).mean()
        ema26 = features['Close'].ewm(span=26, adjust=False).mean()
        features['MACD'] = ema12 - ema26
        features['MACD_Signal'] = features['MACD'].ewm(span=9, adjust=False).mean()
        features['MACD_Hist'] = features['MACD'] - features['MACD_Signal']
        
        # ================================================================
        # BOLLINGER BANDS
        # ================================================================
        for period in [20]:
            sma = features['Close'].rolling(period).mean()
            std = features['Close'].rolling(period).std()
            features[f'BB_Upper_{period}'] = sma + 2 * std
            features[f'BB_Lower_{period}'] = sma - 2 * std
            features[f'BB_Mid_{period}'] = sma
            features[f'BB_Width_{period}'] = (features[f'BB_Upper_{period}'] - features[f'BB_Lower_{period}']) / sma
            features[f'BB_Pct_{period}'] = (features['Close'] - features[f'BB_Lower_{period}']) / \
                                           (features[f'BB_Upper_{period}'] - features[f'BB_Lower_{period}'])
        
        # ================================================================
        # STOCHASTIC OSCILLATOR
        # ================================================================
        for period in [14, 21]:
            low_min = features['Low'].rolling(period).min()
            high_max = features['High'].rolling(period).max()
            features[f'Stoch_K_{period}'] = 100 * (features['Close'] - low_min) / (high_max - low_min)
            features[f'Stoch_D_{period}'] = features[f'Stoch_K_{period}'].rolling(3).mean()
        
        # ================================================================
        # ADX (Average Directional Index)
        # ================================================================
        plus_dm = features['High'].diff()
        minus_dm = -features['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            features['High'] - features['Low'],
            abs(features['High'] - features['Close'].shift(1)),
            abs(features['Low'] - features['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
        features['ADX'] = dx.rolling(14).mean()
        features['Plus_DI'] = plus_di
        features['Minus_DI'] = minus_di
        features['DI_Diff'] = plus_di - minus_di
        
        # ================================================================
        # DONCHIAN CHANNEL
        # ================================================================
        for period in [10, 20, 55]:
            features[f'Donchian_High_{period}'] = features['High'].rolling(period).max()
            features[f'Donchian_Low_{period}'] = features['Low'].rolling(period).min()
            features[f'Donchian_Mid_{period}'] = (features[f'Donchian_High_{period}'] + 
                                                   features[f'Donchian_Low_{period}']) / 2
        
        # ================================================================
        # KELTNER CHANNEL
        # ================================================================
        ema20 = features['EMA_20']
        atr10 = features['ATR_10']
        features['Keltner_Upper'] = ema20 + 2 * atr10
        features['Keltner_Lower'] = ema20 - 2 * atr10
        features['Keltner_Mid'] = ema20
        
        # ================================================================
        # ROC (Rate of Change) & Momentum
        # ================================================================
        for period in [5, 10, 20]:
            features[f'ROC_{period}'] = features['Close'].pct_change(period) * 100
            features[f'Momentum_{period}'] = features['Close'] - features['Close'].shift(period)
        
        # ================================================================
        # Z-SCORE
        # ================================================================
        for period in [20, 50]:
            features[f'ZScore_{period}'] = (features['Close'] - features['Close'].rolling(period).mean()) / \
                                            features['Close'].rolling(period).std()
        
        # ================================================================
        # VOLATILITY MEASURES
        # ================================================================
        features['Volatility_20'] = features['Returns'].rolling(20).std() * np.sqrt(252)
        features['Volatility_50'] = features['Returns'].rolling(50).std() * np.sqrt(252)
        
        # Parkinson Volatility
        features['Parkinson_Vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(features['High'] / features['Low']) ** 2).rolling(20).mean())
        ) * np.sqrt(252)
        
        # ================================================================
        # CCI (Commodity Channel Index)
        # ================================================================
        tp = (features['High'] + features['Low'] + features['Close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        features['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # ================================================================
        # WILLIAMS %R
        # ================================================================
        for period in [14]:
            highest_high = features['High'].rolling(period).max()
            lowest_low = features['Low'].rolling(period).min()
            features[f'Williams_R_{period}'] = -100 * (highest_high - features['Close']) / (highest_high - lowest_low)
        
        # ================================================================
        # SUPERTREND
        # ================================================================
        atr = features['ATR_10']
        multiplier = 3.0
        hl2 = (features['High'] + features['Low']) / 2
        
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=features.index, dtype=float)
        direction = pd.Series(index=features.index, dtype=float)
        
        for i in range(1, len(features)):
            if features['Close'].iloc[i] > upperband.iloc[i-1]:
                direction.iloc[i] = 1
            elif features['Close'].iloc[i] < lowerband.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1] if pd.notna(direction.iloc[i-1]) else 1
                
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lowerband.iloc[i]
            else:
                supertrend.iloc[i] = upperband.iloc[i]
        
        features['Supertrend'] = supertrend
        features['Supertrend_Direction'] = direction
        
        # ================================================================
        # ICHIMOKU CLOUD
        # ================================================================
        # Tenkan-sen (Conversion Line)
        period9_high = features['High'].rolling(9).max()
        period9_low = features['Low'].rolling(9).min()
        features['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = features['High'].rolling(26).max()
        period26_low = features['Low'].rolling(26).min()
        features['Ichimoku_Kijun'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        features['Ichimoku_SpanA'] = ((features['Ichimoku_Tenkan'] + features['Ichimoku_Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        period52_high = features['High'].rolling(52).max()
        period52_low = features['Low'].rolling(52).min()
        features['Ichimoku_SpanB'] = ((period52_high + period52_low) / 2).shift(26)
        
        # ================================================================
        # VWAP (if volume available)
        # ================================================================
        if 'Volume' in features.columns and features['Volume'].sum() > 0:
            tp = (features['High'] + features['Low'] + features['Close']) / 3
            features['VWAP'] = (tp * features['Volume']).cumsum() / features['Volume'].cumsum()
        
        # ================================================================
        # SQUEEZE INDICATOR (Bollinger inside Keltner)
        # ================================================================
        bb_upper = features['BB_Upper_20']
        bb_lower = features['BB_Lower_20']
        kc_upper = features['Keltner_Upper']
        kc_lower = features['Keltner_Lower']
        
        features['Squeeze_On'] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
        
        # ================================================================
        # PIVOT POINTS
        # ================================================================
        features['Pivot'] = (features['High'].shift(1) + features['Low'].shift(1) + features['Close'].shift(1)) / 3
        features['R1'] = 2 * features['Pivot'] - features['Low'].shift(1)
        features['S1'] = 2 * features['Pivot'] - features['High'].shift(1)
        features['R2'] = features['Pivot'] + (features['High'].shift(1) - features['Low'].shift(1))
        features['S2'] = features['Pivot'] - (features['High'].shift(1) - features['Low'].shift(1))
        
        # ================================================================
        # HEIKIN ASHI
        # ================================================================
        features['HA_Close'] = (features['Open'] + features['High'] + features['Low'] + features['Close']) / 4
        features['HA_Open'] = features['HA_Close'].shift(1)  # Simplified
        features['HA_Open'] = (features['HA_Open'].fillna(features['Open']) + features['HA_Close'].shift(1).fillna(features['Close'])) / 2
        features['HA_High'] = features[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        features['HA_Low'] = features[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        # Drop NaN rows (use a reasonable warmup period)
        features = features.iloc[200:].dropna()
        
        self.feature_cache[cache_key] = features
        return features


# ============================================================================
# DATA SPLITTER
# ============================================================================

class DataSplitter:
    """Manages strict temporal data splitting to prevent lookahead bias."""
    
    def __init__(self, train_pct: float = 70.0, val_pct: float = 15.0, test_pct: float = 15.0):
        assert abs(train_pct + val_pct + test_pct - 100.0) < 0.01, "Splits must sum to 100%"
        
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.split_info: Dict[str, Dict] = {}
    
    def get_splits(self, features: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        n = len(features)
        
        train_end = int(n * self.train_pct / 100)
        val_end = int(n * (self.train_pct + self.val_pct) / 100)
        
        train_data = features.iloc[:train_end].copy()
        val_data = features.iloc[train_end:val_end].copy()
        test_data = features.iloc[val_end:].copy()
        
        key = f"{symbol}_{timeframe}"
        self.split_info[key] = {
            'total_bars': n,
            'train_bars': len(train_data),
            'val_bars': len(val_data),
            'test_bars': len(test_data),
            'train_start': str(train_data.index[0]) if len(train_data) > 0 else None,
            'train_end': str(train_data.index[-1]) if len(train_data) > 0 else None,
            'val_start': str(val_data.index[0]) if len(val_data) > 0 else None,
            'val_end': str(val_data.index[-1]) if len(val_data) > 0 else None,
            'test_start': str(test_data.index[0]) if len(test_data) > 0 else None,
            'test_end': str(test_data.index[-1]) if len(test_data) > 0 else None,
        }
        
        return {'train': train_data, 'validation': val_data, 'test': test_data}
    
    def get_train_data(self, features: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        return self.get_splits(features, symbol, timeframe)['train']
    
    def get_val_data(self, features: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        return self.get_splits(features, symbol, timeframe)['validation']
    
    def get_test_data(self, features: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        return self.get_splits(features, symbol, timeframe)['test']
    
    def get_train_val_data(self, features: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        splits = self.get_splits(features, symbol, timeframe)
        return pd.concat([splits['train'], splits['validation']])


# ============================================================================
# BACKTESTER (CORRECTED WITH SPREAD)
# ============================================================================

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10000.0
    position_size_pct: float = 1.0  # Risk per trade as % of equity
    slippage_pips: float = 0.5  # Additional slippage beyond spread
    max_positions: int = 1  # Max concurrent positions
    use_spread: bool = True  # Whether to apply spread costs
    use_commission: bool = True  # Whether to apply commission
    use_swap: bool = False  # Whether to apply overnight swap (for multi-day)


class Backtester:
    """
    Backtester with proper spread modeling.
    
    Key improvements:
    - Spread is applied on both entry and exit (full round-trip cost)
    - Commission is per-lot per round trip
    - Position sizing respects the actual entry price after spread
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
    
    def run(self, strategy, symbol: str, timeframe: str, 
            features: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest and return results."""
        
        if len(features) < 100:
            return self._empty_result()
        
        spec = INSTRUMENT_SPECS.get(symbol, InstrumentSpec(symbol, 4))
        
        # Generate signals
        signals = strategy.generate_signals(features, symbol)
        
        # Simulation state
        equity = self.config.initial_capital
        peak_equity = equity
        max_drawdown = 0.0
        
        in_position = False
        position_direction = 0  # 1 for long, -1 for short
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        position_size = 0.0
        entry_bar = 0
        
        trades = []
        equity_curve = []
        
        warmup = 50
        
        for i in range(warmup, len(features)):
            current_bar = features.iloc[i]
            timestamp = features.index[i]
            
            high = current_bar['High']
            low = current_bar['Low']
            close = current_bar['Close']
            open_price = current_bar['Open']
            
            # ============================================================
            # CHECK EXITS
            # ============================================================
            if in_position:
                exit_price = None
                exit_reason = None
                
                # For checking SL/TP, we need to consider the spread
                # Long positions exit at BID (lower), Short at ASK (higher)
                
                if position_direction == 1:  # Long position
                    # Check if SL hit (worst case: low of bar)
                    bid_low = low - spec.get_half_spread_price() if self.config.use_spread else low
                    bid_high = high - spec.get_half_spread_price() if self.config.use_spread else high
                    
                    if bid_low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'sl'
                    elif bid_high >= take_profit:
                        exit_price = take_profit
                        exit_reason = 'tp'
                        
                else:  # Short position
                    # Check if SL hit (worst case: high of bar)
                    ask_high = high + spec.get_half_spread_price() if self.config.use_spread else high
                    ask_low = low + spec.get_half_spread_price() if self.config.use_spread else low
                    
                    if ask_high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'sl'
                    elif ask_low <= take_profit:
                        exit_price = take_profit
                        exit_reason = 'tp'
                
                if exit_price:
                    # Calculate P&L
                    if position_direction == 1:
                        pnl_pips = spec.price_to_pips(exit_price - entry_price)
                    else:
                        pnl_pips = spec.price_to_pips(entry_price - exit_price)
                    
                    pnl_dollars = pnl_pips * position_size * spec.pip_value
                    
                    # Deduct commission
                    if self.config.use_commission:
                        pnl_dollars -= spec.commission_per_lot * position_size
                    
                    equity += pnl_dollars
                    
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'direction': 'LONG' if position_direction == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pips': round(pnl_pips, 2),
                        'pnl_dollars': round(pnl_dollars, 2),
                        'exit_reason': exit_reason,
                        'position_size': position_size,
                    })
                    
                    in_position = False
            
            # ============================================================
            # CHECK ENTRIES
            # ============================================================
            if not in_position:
                signal = signals.iloc[i]
                
                if signal != 0:
                    direction = int(signal)
                    is_long = direction == 1
                    
                    # Calculate stops using historical data only
                    sl_pips, tp_pips = strategy.calculate_stops(features.iloc[:i+1], direction, symbol)
                    
                    # Entry price: Use OPEN of current bar (signal bar)
                    # Apply spread: Longs at ASK, Shorts at BID
                    if self.config.use_spread:
                        entry_price = spec.get_entry_price(open_price, is_long)
                    else:
                        entry_price = open_price
                    
                    # Apply additional slippage
                    if direction == 1:
                        entry_price += spec.pips_to_price(self.config.slippage_pips)
                    else:
                        entry_price -= spec.pips_to_price(self.config.slippage_pips)
                    
                    # Set SL/TP from entry price
                    if direction == 1:
                        stop_loss = entry_price - spec.pips_to_price(sl_pips)
                        take_profit = entry_price + spec.pips_to_price(tp_pips)
                    else:
                        stop_loss = entry_price + spec.pips_to_price(sl_pips)
                        take_profit = entry_price - spec.pips_to_price(tp_pips)
                    
                    # Position size based on risk
                    risk_amount = equity * (self.config.position_size_pct / 100)
                    if sl_pips > 0 and spec.pip_value > 0:
                        position_size = risk_amount / (sl_pips * spec.pip_value)
                    else:
                        position_size = 0.01
                    
                    # Apply lot size limits
                    position_size = max(spec.min_lot, min(position_size, spec.max_lot))
                    
                    in_position = True
                    position_direction = direction
                    entry_bar = i
            
            # Track equity
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)
        
        # Close any remaining position at last close
        if in_position:
            is_long = position_direction == 1
            if self.config.use_spread:
                exit_price = spec.get_exit_price(features['Close'].iloc[-1], is_long)
            else:
                exit_price = features['Close'].iloc[-1]
            
            if position_direction == 1:
                pnl_pips = spec.price_to_pips(exit_price - entry_price)
            else:
                pnl_pips = spec.price_to_pips(entry_price - exit_price)
            
            pnl_dollars = pnl_pips * position_size * spec.pip_value
            if self.config.use_commission:
                pnl_dollars -= spec.commission_per_lot * position_size
            
            equity += pnl_dollars
            
            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(features) - 1,
                'direction': 'LONG' if position_direction == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pips': round(pnl_pips, 2),
                'pnl_dollars': round(pnl_dollars, 2),
                'exit_reason': 'end_of_data',
                'position_size': position_size,
            })
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity, equity_curve, max_drawdown)
    
    def _calculate_metrics(self, trades: List, final_equity: float, 
                          equity_curve: List, max_dd: float) -> Dict[str, Any]:
        """Calculate performance metrics."""
        
        if not trades:
            return self._empty_result()
        
        wins = [t for t in trades if t['pnl_pips'] > 0]
        losses = [t for t in trades if t['pnl_pips'] < 0]
        
        total_pnl = sum(t['pnl_dollars'] for t in trades)
        win_pnl = sum(t['pnl_dollars'] for t in wins) if wins else 0
        loss_pnl = abs(sum(t['pnl_dollars'] for t in losses)) if losses else 0
        
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else 99.0
        
        avg_win = sum(t['pnl_pips'] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t['pnl_pips'] for t in losses) / len(losses)) if losses else 0
        
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
        
        # Sharpe ratio (annualized, assuming ~252 trading days)
        if len(trades) > 1:
            trade_returns = [t['pnl_dollars'] / self.config.initial_capital for t in trades]
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
        else:
            sharpe = 0
        
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital * 100
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'profit_factor': round(min(profit_factor, 99), 2),
            'sharpe_ratio': round(sharpe, 2),
            'avg_win_pips': round(avg_win, 2),
            'avg_loss_pips': round(avg_loss, 2),
            'expectancy_pips': round(expectancy, 2),
            'final_equity': round(final_equity, 2),
            'trades': trades,
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return_pct': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'avg_win_pips': 0,
            'avg_loss_pips': 0,
            'expectancy_pips': 0,
            'final_equity': self.config.initial_capital,
            'trades': [],
        }
