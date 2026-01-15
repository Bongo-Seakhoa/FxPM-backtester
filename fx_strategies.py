"""
FX Portfolio Manager - Trading Strategies
==========================================

Contains all trading strategy implementations.
Includes: 18 strategies across Trend Following, Mean Reversion, Breakout/Momentum categories.

Version: 2.1
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple

# Import from core
from fx_core import StrategyCategory, INSTRUMENT_SPECS, InstrumentSpec


# ============================================================================
# BASE STRATEGY CLASS
# ============================================================================

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, **params):
        self.params = params
        self._default_params = self.get_default_params()
        for key, value in self._default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def category(self) -> StrategyCategory:
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate trading signals. Returns Series of -1, 0, 1."""
        pass
    
    def calculate_stops(self, features: pd.DataFrame, signal: int, symbol: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit in pips."""
        atr_col = 'ATR_14'
        if atr_col in features.columns:
            atr = features[atr_col].iloc[-1]
        else:
            high_low = features['High'] - features['Low']
            high_close = abs(features['High'] - features['Close'].shift(1))
            low_close = abs(features['Low'] - features['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
        
        spec = INSTRUMENT_SPECS.get(symbol, InstrumentSpec(symbol, 4))
        atr_pips = spec.price_to_pips(atr)
        
        sl_mult = self.params.get('sl_atr_mult', 2.0)
        tp_mult = self.params.get('tp_atr_mult', 3.0)
        
        sl_pips = max(5, atr_pips * sl_mult)
        tp_pips = max(10, atr_pips * tp_mult)
        
        return sl_pips, tp_pips
    
    def get_param_grid(self) -> Dict[str, List]:
        """Return parameter grid for optimization."""
        return {}


# ============================================================================
# TREND FOLLOWING STRATEGIES
# ============================================================================

class EMACrossoverStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "EMACrossoverStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'fast_period': 10, 'slow_period': 20, 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        fast = self.params.get('fast_period', 10)
        slow = self.params.get('slow_period', 20)
        
        fast_ema = features['Close'].ewm(span=fast, adjust=False).mean()
        slow_ema = features['Close'].ewm(span=slow, adjust=False).mean()
        
        signals = pd.Series(0, index=features.index)
        cross_above = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        cross_below = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        signals[cross_above] = 1
        signals[cross_below] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'fast_period': [5, 8, 10, 13], 'slow_period': [20, 21, 30, 50]}


class SupertrendStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "SupertrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'atr_period': 10, 'multiplier': 3.0, 'sl_atr_mult': 1.5, 'tp_atr_mult': 3.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('atr_period', 10)
        mult = self.params.get('multiplier', 3.0)
        
        high_low = features['High'] - features['Low']
        high_close = abs(features['High'] - features['Close'].shift(1))
        low_close = abs(features['Low'] - features['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        hl2 = (features['High'] + features['Low']) / 2
        upperband = hl2 + (mult * atr)
        lowerband = hl2 - (mult * atr)
        
        direction = pd.Series(0, index=features.index)
        for i in range(1, len(features)):
            if features['Close'].iloc[i] > upperband.iloc[i-1]:
                direction.iloc[i] = 1
            elif features['Close'].iloc[i] < lowerband.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
        
        signals = pd.Series(0, index=features.index)
        dir_change = direction.diff()
        signals[dir_change == 2] = 1
        signals[dir_change == -2] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'atr_period': [7, 10, 14], 'multiplier': [2.0, 3.0, 4.0]}


class MACDTrendStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "MACDTrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        fast = self.params.get('fast_period', 12)
        slow = self.params.get('slow_period', 26)
        sig = self.params.get('signal_period', 9)
        
        ema_fast = features['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = features['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=sig, adjust=False).mean()
        
        signals = pd.Series(0, index=features.index)
        cross_above = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        cross_below = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        signals[cross_above] = 1
        signals[cross_below] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'fast_period': [8, 12, 16], 'slow_period': [21, 26, 30]}


class ADXTrendStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ADXTrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'adx_period': 14, 'adx_threshold': 25, 'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        threshold = self.params.get('adx_threshold', 25)
        
        if 'ADX' in features.columns and 'Plus_DI' in features.columns:
            adx = features['ADX']
            plus_di = features['Plus_DI']
            minus_di = features['Minus_DI']
        else:
            period = self.params.get('adx_period', 14)
            plus_dm = features['High'].diff()
            minus_dm = -features['Low'].diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            tr = pd.concat([features['High'] - features['Low'],
                           abs(features['High'] - features['Close'].shift(1)),
                           abs(features['Low'] - features['Close'].shift(1))], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
        
        signals = pd.Series(0, index=features.index)
        strong_trend = adx > threshold
        di_cross_up = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        di_cross_down = (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))
        signals[strong_trend & di_cross_up] = 1
        signals[strong_trend & di_cross_down] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'adx_threshold': [20, 25, 30]}


class IchimokuStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "IchimokuStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'tenkan_period': 9, 'kijun_period': 26, 'sl_atr_mult': 2.5, 'tp_atr_mult': 4.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        if 'Ichimoku_Tenkan' in features.columns:
            tenkan = features['Ichimoku_Tenkan']
            kijun = features['Ichimoku_Kijun']
            span_a = features['Ichimoku_SpanA']
            span_b = features['Ichimoku_SpanB']
        else:
            tenkan_p = self.params.get('tenkan_period', 9)
            kijun_p = self.params.get('kijun_period', 26)
            tenkan = (features['High'].rolling(tenkan_p).max() + features['Low'].rolling(tenkan_p).min()) / 2
            kijun = (features['High'].rolling(kijun_p).max() + features['Low'].rolling(kijun_p).min()) / 2
            span_a = ((tenkan + kijun) / 2).shift(kijun_p)
            span_b = ((features['High'].rolling(52).max() + features['Low'].rolling(52).min()) / 2).shift(kijun_p)
        
        signals = pd.Series(0, index=features.index)
        close = features['Close']
        
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        
        above_cloud = close > cloud_top
        below_cloud = close < cloud_bottom
        tk_cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        tk_cross_down = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
        
        signals[above_cloud & tk_cross_up] = 1
        signals[below_cloud & tk_cross_down] = -1
        return signals


class HullMATrendStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "HullMATrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'period': 20, 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.5}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 20)
        
        if 'HMA_20' in features.columns and period == 20:
            hma = features['HMA_20']
        else:
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            def wma(series, p):
                weights = np.arange(1, p + 1)
                return series.rolling(p).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
            
            wma_half = wma(features['Close'], half_period)
            wma_full = wma(features['Close'], period)
            raw_hma = 2 * wma_half - wma_full
            hma = wma(raw_hma, sqrt_period)
        
        signals = pd.Series(0, index=features.index)
        hma_direction = np.sign(hma.diff())
        direction_change = hma_direction.diff()
        signals[direction_change > 0] = 1
        signals[direction_change < 0] = -1
        return signals


# ============================================================================
# MEAN REVERSION STRATEGIES
# ============================================================================

class RSIExtremesStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "RSIExtremesStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'rsi_period': 14, 'overbought': 70, 'oversold': 30, 'sl_atr_mult': 1.5, 'tp_atr_mult': 2.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('rsi_period', 14)
        ob = self.params.get('overbought', 70)
        os_level = self.params.get('oversold', 30)
        
        rsi_col = f'RSI_{period}'
        if rsi_col in features.columns:
            rsi = features[rsi_col]
        else:
            delta = features['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=features.index)
        was_oversold = rsi.shift(1) < os_level
        crosses_above = rsi >= os_level
        signals[was_oversold & crosses_above] = 1
        
        was_overbought = rsi.shift(1) > ob
        crosses_below = rsi <= ob
        signals[was_overbought & crosses_below] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'rsi_period': [7, 14, 21], 'overbought': [70, 75], 'oversold': [25, 30]}


class BollingerBounceStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "BollingerBounceStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'bb_period': 20, 'bb_std': 2.0, 'sl_atr_mult': 1.5, 'tp_atr_mult': 2.5}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('bb_period', 20)
        
        if f'BB_Upper_{period}' in features.columns:
            upper = features[f'BB_Upper_{period}']
            lower = features[f'BB_Lower_{period}']
        else:
            std_mult = self.params.get('bb_std', 2.0)
            sma = features['Close'].rolling(period).mean()
            std = features['Close'].rolling(period).std()
            upper = sma + std_mult * std
            lower = sma - std_mult * std
        
        signals = pd.Series(0, index=features.index)
        touch_lower = features['Low'] <= lower
        bullish_close = features['Close'] > features['Open']
        signals[touch_lower & bullish_close] = 1
        
        touch_upper = features['High'] >= upper
        bearish_close = features['Close'] < features['Open']
        signals[touch_upper & bearish_close] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'bb_period': [15, 20, 25]}


class ZScoreMRStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "ZScoreMRStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'zscore_period': 20, 'entry_threshold': 2.0, 'sl_atr_mult': 1.5, 'tp_atr_mult': 2.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('zscore_period', 20)
        threshold = self.params.get('entry_threshold', 2.0)
        
        if f'ZScore_{period}' in features.columns:
            zscore = features[f'ZScore_{period}']
        else:
            mean = features['Close'].rolling(period).mean()
            std = features['Close'].rolling(period).std()
            zscore = (features['Close'] - mean) / std
        
        signals = pd.Series(0, index=features.index)
        was_below = zscore.shift(1) < -threshold
        crosses_up = zscore >= -threshold
        signals[was_below & crosses_up] = 1
        
        was_above = zscore.shift(1) > threshold
        crosses_down = zscore <= threshold
        signals[was_above & crosses_down] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'zscore_period': [15, 20, 30], 'entry_threshold': [1.5, 2.0, 2.5]}


class StochasticReversalStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "StochasticReversalStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'k_period': 14, 'd_period': 3, 'overbought': 80, 'oversold': 20, 'sl_atr_mult': 1.5, 'tp_atr_mult': 2.5}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        k_period = self.params.get('k_period', 14)
        ob = self.params.get('overbought', 80)
        os_level = self.params.get('oversold', 20)
        
        if f'Stoch_K_{k_period}' in features.columns:
            k = features[f'Stoch_K_{k_period}']
            d = features[f'Stoch_D_{k_period}']
        else:
            d_period = self.params.get('d_period', 3)
            low_min = features['Low'].rolling(k_period).min()
            high_max = features['High'].rolling(k_period).max()
            k = 100 * (features['Close'] - low_min) / (high_max - low_min)
            d = k.rolling(d_period).mean()
        
        signals = pd.Series(0, index=features.index)
        k_cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        in_oversold = (k < os_level) | (k.shift(1) < os_level)
        signals[k_cross_up & in_oversold] = 1
        
        k_cross_down = (k < d) & (k.shift(1) >= d.shift(1))
        in_overbought = (k > ob) | (k.shift(1) > ob)
        signals[k_cross_down & in_overbought] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'k_period': [9, 14, 21], 'overbought': [75, 80], 'oversold': [20, 25]}


class CCIReversalStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "CCIReversalStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'cci_period': 20, 'overbought': 100, 'oversold': -100, 'sl_atr_mult': 1.5, 'tp_atr_mult': 2.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        ob = self.params.get('overbought', 100)
        os_level = self.params.get('oversold', -100)
        
        if 'CCI' in features.columns:
            cci = features['CCI']
        else:
            period = self.params.get('cci_period', 20)
            tp = (features['High'] + features['Low'] + features['Close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            cci = (tp - sma_tp) / (0.015 * mad)
        
        signals = pd.Series(0, index=features.index)
        was_oversold = cci.shift(1) < os_level
        crosses_up = cci >= os_level
        signals[was_oversold & crosses_up] = 1
        
        was_overbought = cci.shift(1) > ob
        crosses_down = cci <= ob
        signals[was_overbought & crosses_down] = -1
        return signals


class WilliamsRStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "WilliamsRStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'period': 14, 'overbought': -20, 'oversold': -80, 'sl_atr_mult': 1.5, 'tp_atr_mult': 2.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 14)
        ob = self.params.get('overbought', -20)
        os_level = self.params.get('oversold', -80)
        
        if f'Williams_R_{period}' in features.columns:
            wr = features[f'Williams_R_{period}']
        else:
            highest_high = features['High'].rolling(period).max()
            lowest_low = features['Low'].rolling(period).min()
            wr = -100 * (highest_high - features['Close']) / (highest_high - lowest_low)
        
        signals = pd.Series(0, index=features.index)
        was_oversold = wr.shift(1) < os_level
        crosses_up = wr >= os_level
        signals[was_oversold & crosses_up] = 1
        
        was_overbought = wr.shift(1) > ob
        crosses_down = wr <= ob
        signals[was_overbought & crosses_down] = -1
        return signals


# ============================================================================
# BREAKOUT / MOMENTUM STRATEGIES
# ============================================================================

class DonchianBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "DonchianBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'entry_period': 20, 'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('entry_period', 20)
        
        if f'Donchian_High_{period}' in features.columns:
            upper = features[f'Donchian_High_{period}'].shift(1)
            lower = features[f'Donchian_Low_{period}'].shift(1)
        else:
            upper = features['High'].rolling(period).max().shift(1)
            lower = features['Low'].rolling(period).min().shift(1)
        
        signals = pd.Series(0, index=features.index)
        signals[features['Close'] > upper] = 1
        signals[features['Close'] < lower] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'entry_period': [10, 20, 30, 55]}


class VolatilityBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "VolatilityBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14, 'breakout_mult': 1.5, 'sl_atr_mult': 1.5, 'tp_atr_mult': 3.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('atr_period', 14)
        mult = self.params.get('breakout_mult', 1.5)
        
        if f'ATR_{period}' in features.columns:
            atr = features[f'ATR_{period}']
        else:
            high_low = features['High'] - features['Low']
            high_close = abs(features['High'] - features['Close'].shift(1))
            low_close = abs(features['Low'] - features['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
        
        prev_close = features['Close'].shift(1)
        move = features['Close'] - prev_close
        threshold = atr * mult
        
        signals = pd.Series(0, index=features.index)
        signals[move > threshold] = 1
        signals[move < -threshold] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'atr_period': [10, 14, 20], 'breakout_mult': [1.0, 1.5, 2.0]}


class MomentumBurstStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "MomentumBurstStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'roc_period': 10, 'roc_threshold': 0.5, 'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('roc_period', 10)
        threshold = self.params.get('roc_threshold', 0.5)
        
        if f'ROC_{period}' in features.columns:
            roc = features[f'ROC_{period}']
        else:
            roc = features['Close'].pct_change(period) * 100
        
        roc_accel = roc - roc.shift(1)
        
        signals = pd.Series(0, index=features.index)
        signals[(roc > threshold) & (roc_accel > 0)] = 1
        signals[(roc < -threshold) & (roc_accel < 0)] = -1
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {'roc_period': [5, 10, 15], 'roc_threshold': [0.3, 0.5, 1.0]}


class SqueezeBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "SqueezeBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'bb_period': 20, 'bb_std': 2.0, 'kc_mult': 1.5, 'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        if 'Squeeze_On' in features.columns:
            squeeze_on = features['Squeeze_On'].astype(bool)
        else:
            bb_period = self.params.get('bb_period', 20)
            bb_std = self.params.get('bb_std', 2.0)
            kc_mult = self.params.get('kc_mult', 1.5)
            
            sma = features['Close'].rolling(bb_period).mean()
            std = features['Close'].rolling(bb_period).std()
            bb_upper = sma + bb_std * std
            bb_lower = sma - bb_std * std
            
            atr = features['ATR_14'] if 'ATR_14' in features.columns else (features['High'] - features['Low']).rolling(14).mean()
            ema = features['Close'].ewm(span=20, adjust=False).mean()
            kc_upper = ema + kc_mult * atr
            kc_lower = ema - kc_mult * atr
            
            squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        squeeze_release = squeeze_on.shift(1) & ~squeeze_on
        
        if 'Momentum_20' in features.columns:
            momentum = features['Momentum_20']
        else:
            momentum = features['Close'] - features['Close'].shift(20)
        
        signals = pd.Series(0, index=features.index)
        signals[squeeze_release & (momentum > 0)] = 1
        signals[squeeze_release & (momentum < 0)] = -1
        return signals


class KeltnerBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "KeltnerBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'ema_period': 20, 'atr_mult': 2.0, 'sl_atr_mult': 1.5, 'tp_atr_mult': 3.0}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        if 'Keltner_Upper' in features.columns:
            upper = features['Keltner_Upper']
            lower = features['Keltner_Lower']
        else:
            ema_period = self.params.get('ema_period', 20)
            atr_mult = self.params.get('atr_mult', 2.0)
            
            ema = features['Close'].ewm(span=ema_period, adjust=False).mean()
            atr = features['ATR_14'] if 'ATR_14' in features.columns else (features['High'] - features['Low']).rolling(14).mean()
            upper = ema + atr_mult * atr
            lower = ema - atr_mult * atr
        
        signals = pd.Series(0, index=features.index)
        signals[features['Close'] > upper] = 1
        signals[features['Close'] < lower] = -1
        return signals


class PivotBreakoutStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "PivotBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'sl_atr_mult': 1.5, 'tp_atr_mult': 2.5}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        if 'R1' in features.columns:
            r1 = features['R1']
            s1 = features['S1']
        else:
            pivot = (features['High'].shift(1) + features['Low'].shift(1) + features['Close'].shift(1)) / 3
            r1 = 2 * pivot - features['Low'].shift(1)
            s1 = 2 * pivot - features['High'].shift(1)
        
        signals = pd.Series(0, index=features.index)
        break_r1 = (features['Close'] > r1) & (features['Close'].shift(1) <= r1.shift(1))
        break_s1 = (features['Close'] < s1) & (features['Close'].shift(1) >= s1.shift(1))
        signals[break_r1] = 1
        signals[break_s1] = -1
        return signals


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

class StrategyRegistry:
    """Registry for all available strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, type] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register all default strategies."""
        strategies = [
            # Trend Following (6)
            EMACrossoverStrategy,
            SupertrendStrategy,
            MACDTrendStrategy,
            ADXTrendStrategy,
            IchimokuStrategy,
            HullMATrendStrategy,
            
            # Mean Reversion (6)
            RSIExtremesStrategy,
            BollingerBounceStrategy,
            ZScoreMRStrategy,
            StochasticReversalStrategy,
            CCIReversalStrategy,
            WilliamsRStrategy,
            
            # Breakout/Momentum (6)
            DonchianBreakoutStrategy,
            VolatilityBreakoutStrategy,
            MomentumBurstStrategy,
            SqueezeBreakoutStrategy,
            KeltnerBreakoutStrategy,
            PivotBreakoutStrategy,
        ]
        
        for strat_class in strategies:
            instance = strat_class()
            self._strategies[instance.name] = strat_class
    
    def get(self, name: str):
        """Get strategy instance by name."""
        if name in self._strategies:
            return self._strategies[name]()
        return None
    
    def get_all(self) -> List:
        """Get all registered strategies."""
        return [cls() for cls in self._strategies.values()]
    
    def create_instance(self, name: str, **params):
        """Create strategy instance with custom parameters."""
        if name in self._strategies:
            return self._strategies[name](**params)
        return None
    
    def list_strategies(self) -> List[str]:
        """List all strategy names."""
        return list(self._strategies.keys())
    
    def get_by_category(self, category: StrategyCategory) -> List:
        """Get all strategies of a specific category."""
        return [cls() for cls in self._strategies.values() if cls().category == category]
