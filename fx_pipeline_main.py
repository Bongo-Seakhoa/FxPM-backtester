"""
FX Portfolio Manager - Production Pipeline
==========================================

Complete 4-stage production pipeline for FX strategy development:
- Process 1: Strategy Application (TRAINING data only)
- Process 2: Strategy Evaluation (TRAINING data only)
- Process 3: Hyperparameter Optimization (TRAINING + VALIDATION)
- Process 4: Live Simulation (TEST data - never seen before)

Usage:
    python fx_pipeline_main.py --data-dir ./data --output-dir ./outputs

Version: 2.1 (Corrected with spread, commission, and proper pip values)
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from itertools import product
import time

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from fx_core import (
    PipelineConfig, Timer, DataLoader, FeatureComputer, DataSplitter,
    Backtester, BacktestConfig, INSTRUMENT_SPECS, InstrumentSpec,
    StrategyCategory
)
from fx_strategies import StrategyRegistry, BaseStrategy


# ============================================================================
# PROCESS 1: STRATEGY APPLICATION
# ============================================================================

class Process1_StrategyApplication:
    """Apply all strategies to all symbols on TRAINING data only."""
    
    def __init__(self, config: PipelineConfig, data_loader: DataLoader,
                 feature_computer: FeatureComputer, data_splitter: DataSplitter,
                 strategy_registry: StrategyRegistry):
        self.config = config
        self.data_loader = data_loader
        self.features = feature_computer
        self.splitter = data_splitter
        self.registry = strategy_registry
        
        self.output_dir = config.output_dir / 'process1'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        symbols = list(self.data_loader.raw_data.keys())
        strategies = self.registry.get_all()
        
        if verbose:
            print("=" * 70)
            print("PROCESS 1: STRATEGY APPLICATION (TRAINING DATA ONLY)")
            print("=" * 70)
            print(f"\nSymbols: {len(symbols)}, Strategies: {len(strategies)}")
        
        start_time = time.time()
        completed = 0
        
        for symbol in symbols:
            for tf in self.config.timeframes:
                full_features = self.features.compute_all_features(symbol, tf)
                if full_features is None or len(full_features) < 200:
                    continue
                
                train_features = self.splitter.get_train_data(full_features, symbol, tf)
                if len(train_features) < 100:
                    continue
                
                for strategy in strategies:
                    try:
                        signals = strategy.generate_signals(train_features, symbol)
                        long_signals = int((signals == 1).sum())
                        short_signals = int((signals == -1).sum())
                        
                        self.results.append({
                            'symbol': symbol, 'timeframe': tf, 'strategy': strategy.name,
                            'category': strategy.category.value, 'train_bars': len(train_features),
                            'long_signals': long_signals, 'short_signals': short_signals,
                            'total_signals': long_signals + short_signals,
                        })
                        completed += 1
                    except Exception as e:
                        pass
                
                if verbose:
                    print(f"  ✅ {symbol} {tf}: {len(train_features)} bars")
        
        elapsed = time.time() - start_time
        summary = {'process': 'Strategy Application', 'completed': completed, 'time_sec': round(elapsed, 2)}
        self._export_results(summary)
        
        if verbose:
            print(f"\n✅ Completed: {completed} combinations in {elapsed:.1f}s")
        return summary
    
    def _export_results(self, summary: Dict) -> None:
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        if self.results:
            pd.DataFrame(self.results).to_csv(self.output_dir / 'applications.csv', index=False)


# ============================================================================
# PROCESS 2: STRATEGY EVALUATION
# ============================================================================

class Process2_StrategyEvaluation:
    """Evaluate all strategies on TRAINING data and find best per symbol."""
    
    def __init__(self, config: PipelineConfig, data_loader: DataLoader,
                 feature_computer: FeatureComputer, data_splitter: DataSplitter,
                 strategy_registry: StrategyRegistry):
        self.config = config
        self.data_loader = data_loader
        self.features = feature_computer
        self.splitter = data_splitter
        self.registry = strategy_registry
        
        self.output_dir = config.output_dir / 'process2'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.backtester = Backtester(BacktestConfig(
            initial_capital=config.initial_capital,
            position_size_pct=config.risk_per_trade_pct,
            slippage_pips=config.slippage_pips,
            use_spread=True, use_commission=True,
        ))
        
        self.evaluation_results = []
        self.best_per_symbol = {}
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        symbols = list(self.data_loader.raw_data.keys())
        strategies = self.registry.get_all()
        
        if verbose:
            print("=" * 70)
            print("PROCESS 2: STRATEGY EVALUATION (TRAINING DATA ONLY)")
            print("=" * 70)
        
        start_time = time.time()
        
        for symbol in symbols:
            if verbose:
                print(f"\n📊 Evaluating {symbol}...")
            
            best_for_symbol = None
            best_score = -999
            
            for tf in self.config.timeframes:
                full_features = self.features.compute_all_features(symbol, tf)
                if full_features is None or len(full_features) < 300:
                    continue
                
                train_features = self.splitter.get_train_data(full_features, symbol, tf)
                if len(train_features) < 150:
                    continue
                
                for strategy in strategies:
                    try:
                        result = self.backtester.run(strategy, symbol, tf, train_features)
                        
                        if result['total_trades'] < self.config.min_trades:
                            continue
                        
                        eval_result = {
                            'symbol': symbol, 'timeframe': tf, 'strategy': strategy.name,
                            'category': strategy.category.value,
                            'total_trades': result['total_trades'],
                            'win_rate': result['win_rate'],
                            'total_return_pct': result['total_return_pct'],
                            'max_drawdown_pct': result['max_drawdown_pct'],
                            'profit_factor': result['profit_factor'],
                            'sharpe_ratio': result['sharpe_ratio'],
                            'expectancy_pips': result['expectancy_pips'],
                        }
                        
                        score = self._calculate_score(result)
                        eval_result['composite_score'] = round(score, 2)
                        self.evaluation_results.append(eval_result)
                        
                        if score > best_score:
                            best_score = score
                            best_for_symbol = eval_result.copy()
                    except:
                        pass
            
            if best_for_symbol:
                self.best_per_symbol[symbol] = best_for_symbol
                if verbose:
                    print(f"   Best: {best_for_symbol['strategy']} @ {best_for_symbol['timeframe']} "
                          f"(Score={best_for_symbol['composite_score']:.1f})")
        
        elapsed = time.time() - start_time
        summary = {
            'process': 'Strategy Evaluation',
            'evaluations': len(self.evaluation_results),
            'recommendations': len(self.best_per_symbol),
            'time_sec': round(elapsed, 2),
        }
        self._export_results(summary)
        
        if verbose:
            self._print_summary()
        return summary
    
    def _calculate_score(self, result: Dict) -> float:
        sharpe = min(max(result.get('sharpe_ratio', 0), -2), 5)
        pf = min(max(result.get('profit_factor', 0), 0), 10)
        win_rate = result.get('win_rate', 0) / 100
        ret = result.get('total_return_pct', 0)
        dd = result.get('max_drawdown_pct', 100)
        
        return_dd = ret / dd if dd > 0 else ret
        return_dd = min(max(return_dd, -5), 20)
        
        score = ((sharpe + 2) / 7 * 25 + pf / 10 * 20 + win_rate * 15 + 
                 (return_dd + 5) / 25 * 25 + min(result.get('expectancy_pips', 0) / 10, 1) * 15)
        
        if dd > 30: score *= 0.6
        elif dd > 25: score *= 0.75
        elif dd > 20: score *= 0.9
        
        return score
    
    def _export_results(self, summary: Dict) -> None:
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        if self.evaluation_results:
            pd.DataFrame(self.evaluation_results).to_csv(self.output_dir / 'all_evaluations.csv', index=False)
        if self.best_per_symbol:
            with open(self.output_dir / 'recommendations.json', 'w') as f:
                json.dump(self.best_per_symbol, f, indent=2)
            pd.DataFrame(list(self.best_per_symbol.values())).to_csv(
                self.output_dir / 'recommendations.csv', index=False)
    
    def _print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("BEST STRATEGY PER SYMBOL")
        print("=" * 70)
        print(f"\n{'Symbol':<10} {'TF':<5} {'Strategy':<26} {'Score':>7} {'Return%':>10}")
        print("-" * 65)
        for r in sorted(self.best_per_symbol.values(), key=lambda x: x['composite_score'], reverse=True):
            print(f"{r['symbol']:<10} {r['timeframe']:<5} {r['strategy'][:26]:<26} "
                  f"{r['composite_score']:>7.1f} {r['total_return_pct']:>+10.1f}")


# ============================================================================
# PROCESS 3: HYPERPARAMETER OPTIMIZATION
# ============================================================================

class Process3_Optimization:
    """Optimize hyperparameters on TRAINING, validate on VALIDATION data."""
    
    def __init__(self, config: PipelineConfig, data_loader: DataLoader,
                 feature_computer: FeatureComputer, data_splitter: DataSplitter,
                 strategy_registry: StrategyRegistry):
        self.config = config
        self.data_loader = data_loader
        self.features = feature_computer
        self.splitter = data_splitter
        self.registry = strategy_registry
        
        self.output_dir = config.output_dir / 'process3'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.recommendations = self._load_recommendations()
        self.optimization_results = {}
    
    def _load_recommendations(self) -> Dict:
        rec_file = self.config.output_dir / 'process2' / 'recommendations.json'
        if rec_file.exists():
            with open(rec_file, 'r') as f:
                return json.load(f)
        return {}
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        if not self.recommendations:
            print("❌ No recommendations found. Run Process 2 first.")
            return {}
        
        if verbose:
            print("=" * 70)
            print("PROCESS 3: HYPERPARAMETER OPTIMIZATION")
            print("=" * 70)
            print(f"\nSymbols to optimize: {len(self.recommendations)}")
        
        start_time = time.time()
        
        bt_config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            position_size_pct=self.config.risk_per_trade_pct,
            slippage_pips=self.config.slippage_pips,
            use_spread=True, use_commission=True,
        )
        backtester = Backtester(bt_config)
        
        for symbol, rec in self.recommendations.items():
            strategy_name = rec['strategy']
            timeframe = rec['timeframe']
            
            if verbose:
                print(f"\n🔧 Optimizing {symbol} ({strategy_name})...")
            
            try:
                full_features = self.features.compute_all_features(symbol, timeframe)
                if full_features is None:
                    continue
                
                train_features = self.splitter.get_train_data(full_features, symbol, timeframe)
                val_features = self.splitter.get_val_data(full_features, symbol, timeframe)
                
                base_strategy = self.registry.get(strategy_name)
                if base_strategy is None:
                    continue
                
                default_params = base_strategy.get_default_params()
                param_grid = base_strategy.get_param_grid()
                
                best_params = default_params.copy()
                best_score = self._calc_score(backtester.run(base_strategy, symbol, timeframe, train_features))
                
                if param_grid:
                    param_names = list(param_grid.keys())
                    param_values = list(param_grid.values())
                    all_combos = list(product(*param_values))
                    
                    if len(all_combos) > self.config.max_param_combos:
                        np.random.seed(42)
                        indices = np.random.choice(len(all_combos), self.config.max_param_combos, replace=False)
                        all_combos = [all_combos[i] for i in indices]
                    
                    for combo in all_combos:
                        test_params = {**default_params, **dict(zip(param_names, combo))}
                        try:
                            test_strategy = self.registry.create_instance(strategy_name, **test_params)
                            result = backtester.run(test_strategy, symbol, timeframe, train_features)
                            if result['total_trades'] >= 10:
                                score = self._calc_score(result)
                                if score > best_score:
                                    best_score = score
                                    best_params = test_params
                        except:
                            pass
                
                # Validate on VALIDATION data
                opt_strategy = self.registry.create_instance(strategy_name, **best_params)
                train_result = backtester.run(opt_strategy, symbol, timeframe, train_features)
                
                if len(val_features) >= 50:
                    val_result = backtester.run(opt_strategy, symbol, timeframe, val_features)
                    robustness = (val_result['total_return_pct'] / abs(train_result['total_return_pct'])
                                 if train_result['total_return_pct'] != 0 else 0)
                else:
                    val_result = {'total_return_pct': 0, 'sharpe_ratio': 0, 'max_drawdown_pct': 0, 'total_trades': 0}
                    robustness = 0
                
                is_validated = ((robustness >= self.config.min_robustness or val_result.get('sharpe_ratio', 0) > 0.3)
                               and val_result.get('max_drawdown_pct', 100) < 35
                               and val_result.get('total_trades', 0) >= 5)
                
                # Convert numpy types
                clean_params = {k: float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v
                               for k, v in best_params.items()}
                
                self.optimization_results[symbol] = {
                    'symbol': symbol, 'timeframe': timeframe, 'strategy': strategy_name,
                    'optimized_params': clean_params,
                    'train_return': float(train_result['total_return_pct']),
                    'train_sharpe': float(train_result['sharpe_ratio']),
                    'val_return': float(val_result.get('total_return_pct', 0)),
                    'val_sharpe': float(val_result.get('sharpe_ratio', 0)),
                    'robustness_ratio': round(float(robustness), 3),
                    'is_validated': bool(is_validated),
                }
                
                if verbose:
                    print(f"   Train: {train_result['total_return_pct']:+.1f}%, Val: {val_result.get('total_return_pct', 0):+.1f}%")
                    print(f"   Robustness: {robustness:.2f} ({'✅' if is_validated else '⚠️'})")
                
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error: {e}")
        
        elapsed = time.time() - start_time
        validated_count = sum(1 for r in self.optimization_results.values() if r.get('is_validated', False))
        
        summary = {
            'process': 'Hyperparameter Optimization',
            'symbols_optimized': len(self.optimization_results),
            'symbols_validated': validated_count,
            'time_sec': round(elapsed, 2),
        }
        self._export_results(summary)
        
        if verbose:
            print(f"\n✅ Validated for Process 4: {validated_count}/{len(self.optimization_results)}")
        return summary
    
    def _calc_score(self, result: Dict) -> float:
        sharpe = max(-2, min(5, result.get('sharpe_ratio', 0)))
        ret = result.get('total_return_pct', 0)
        dd = result.get('max_drawdown_pct', 100)
        dd_penalty = 1.0 if dd < 15 else 0.8 if dd < 25 else 0.5
        return (sharpe * 30 + min(ret, 500) * 0.1) * dd_penalty
    
    def _export_results(self, summary: Dict) -> None:
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.optimization_results:
            rows = [{k: v for k, v in r.items() if k != 'optimized_params'} 
                    for r in self.optimization_results.values()]
            pd.DataFrame(rows).to_csv(self.output_dir / 'optimization_results.csv', index=False)
            
            with open(self.output_dir / 'all_configs.json', 'w') as f:
                json.dump(self.optimization_results, f, indent=2)
            
            validated = {k: v for k, v in self.optimization_results.items() if v.get('is_validated', False)}
            with open(self.output_dir / 'validated_configs.json', 'w') as f:
                json.dump(validated, f, indent=2)


# ============================================================================
# PROCESS 4: LIVE SIMULATION
# ============================================================================

class Process4_LiveSimulation:
    """Bar-by-bar simulation on completely UNSEEN test data."""
    
    def __init__(self, config: PipelineConfig, data_loader: DataLoader,
                 feature_computer: FeatureComputer, data_splitter: DataSplitter,
                 strategy_registry: StrategyRegistry):
        self.config = config
        self.data_loader = data_loader
        self.features = feature_computer
        self.splitter = data_splitter
        self.registry = strategy_registry
        
        self.output_dir = config.output_dir / 'process4'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs = self._load_configs()
        self.simulation_results = {}
    
    def _load_configs(self) -> Dict:
        for filename in ['validated_configs.json', 'all_configs.json']:
            filepath = self.config.output_dir / 'process3' / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    configs = json.load(f)
                    if configs:
                        print(f"✅ Loaded {len(configs)} configs from {filename}")
                        return configs
        
        rec_file = self.config.output_dir / 'process2' / 'recommendations.json'
        if rec_file.exists():
            with open(rec_file, 'r') as f:
                recs = json.load(f)
                return {s: {'symbol': s, 'timeframe': r['timeframe'], 'strategy': r['strategy'], 
                           'optimized_params': {}} for s, r in recs.items()}
        return {}
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        if not self.configs:
            print("❌ No configurations found. Run Process 2 or 3 first.")
            return {}
        
        if verbose:
            print("=" * 70)
            print("PROCESS 4: LIVE SIMULATION (UNSEEN TEST DATA)")
            print("=" * 70)
            print(f"\n🔒 Using TEST data (last {self.config.test_pct}%) - NEVER seen before!")
            print(f"\nSymbols to simulate: {len(self.configs)}")
        
        start_time = time.time()
        
        for symbol, config in self.configs.items():
            if verbose:
                print(f"\n🔄 Simulating {symbol}...")
            
            try:
                result = self._simulate_symbol(symbol, config, verbose)
                self.simulation_results[symbol] = result
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error: {e}")
        
        elapsed = time.time() - start_time
        summary = self._generate_report(elapsed, verbose)
        self._export_results(summary)
        
        return summary
    
    def _simulate_symbol(self, symbol: str, config: Dict, verbose: bool) -> Dict:
        timeframe = config.get('timeframe', 'H4')
        strategy_name = config.get('strategy', 'DonchianBreakoutStrategy')
        params = config.get('optimized_params', {})
        
        full_features = self.features.compute_all_features(symbol, timeframe)
        if full_features is None:
            raise ValueError("No features available")
        
        test_features = self.splitter.get_test_data(full_features, symbol, timeframe)
        if len(test_features) < 50:
            raise ValueError(f"Insufficient test data: {len(test_features)} bars")
        
        if verbose:
            print(f"   TEST data: {len(test_features)} bars")
        
        strategy = self.registry.create_instance(strategy_name, **params)
        if strategy is None:
            strategy = self.registry.get(strategy_name)
        
        spec = INSTRUMENT_SPECS.get(symbol, InstrumentSpec(symbol, 4))
        
        # Simulation state
        equity = self.config.initial_capital
        peak_equity = equity
        in_position = False
        position_direction = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        position_size = 0.0
        
        trades = []
        max_dd = 0.0
        warmup = 50
        
        for i in range(warmup, len(test_features)):
            current_bar = test_features.iloc[i]
            timestamp = test_features.index[i]
            
            high, low, close, open_price = current_bar['High'], current_bar['Low'], current_bar['Close'], current_bar['Open']
            
            # Check exits
            if in_position:
                exit_price = None
                exit_reason = None
                
                if position_direction == 1:  # Long - exit at BID
                    bid_low = low - spec.get_half_spread_price()
                    bid_high = high - spec.get_half_spread_price()
                    if bid_low <= stop_loss:
                        exit_price, exit_reason = stop_loss, 'sl'
                    elif bid_high >= take_profit:
                        exit_price, exit_reason = take_profit, 'tp'
                else:  # Short - exit at ASK
                    ask_high = high + spec.get_half_spread_price()
                    ask_low = low + spec.get_half_spread_price()
                    if ask_high >= stop_loss:
                        exit_price, exit_reason = stop_loss, 'sl'
                    elif ask_low <= take_profit:
                        exit_price, exit_reason = take_profit, 'tp'
                
                if exit_price:
                    pnl_pips = spec.price_to_pips(exit_price - entry_price) if position_direction == 1 else spec.price_to_pips(entry_price - exit_price)
                    pnl_dollars = pnl_pips * position_size * spec.pip_value - spec.commission_per_lot * position_size
                    equity += pnl_dollars
                    
                    trades.append({
                        'exit_time': str(timestamp), 'direction': 'LONG' if position_direction == 1 else 'SHORT',
                        'entry_price': round(entry_price, 5), 'exit_price': round(exit_price, 5),
                        'pnl_pips': round(pnl_pips, 2), 'pnl_dollars': round(pnl_dollars, 2), 'reason': exit_reason,
                    })
                    in_position = False
            
            # Check entries
            if not in_position:
                try:
                    history = test_features.iloc[:i+1]
                    signals = strategy.generate_signals(history, symbol)
                    current_signal = signals.iloc[-1]
                    
                    if current_signal != 0:
                        direction = int(current_signal)
                        is_long = direction == 1
                        sl_pips, tp_pips = strategy.calculate_stops(history, direction, symbol)
                        
                        entry_price = spec.get_entry_price(open_price, is_long)
                        entry_price += spec.pips_to_price(self.config.slippage_pips) * (1 if is_long else -1)
                        
                        if direction == 1:
                            stop_loss = entry_price - spec.pips_to_price(sl_pips)
                            take_profit = entry_price + spec.pips_to_price(tp_pips)
                        else:
                            stop_loss = entry_price + spec.pips_to_price(sl_pips)
                            take_profit = entry_price - spec.pips_to_price(tp_pips)
                        
                        risk_amount = equity * (self.config.risk_per_trade_pct / 100)
                        position_size = risk_amount / (sl_pips * spec.pip_value) if sl_pips > 0 else 0.01
                        position_size = max(spec.min_lot, min(position_size, spec.max_lot))
                        
                        in_position = True
                        position_direction = direction
                except:
                    pass
            
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)
        
        # Close remaining position
        if in_position:
            is_long = position_direction == 1
            exit_price = spec.get_exit_price(test_features['Close'].iloc[-1], is_long)
            pnl_pips = spec.price_to_pips(exit_price - entry_price) if position_direction == 1 else spec.price_to_pips(entry_price - exit_price)
            pnl_dollars = pnl_pips * position_size * spec.pip_value - spec.commission_per_lot * position_size
            equity += pnl_dollars
            trades.append({
                'exit_time': str(test_features.index[-1]), 'direction': 'LONG' if position_direction == 1 else 'SHORT',
                'entry_price': round(entry_price, 5), 'exit_price': round(exit_price, 5),
                'pnl_pips': round(pnl_pips, 2), 'pnl_dollars': round(pnl_dollars, 2), 'reason': 'end_of_test',
            })
        
        metrics = self._calculate_metrics(trades, equity, max_dd)
        
        if verbose:
            print(f"   ✅ {metrics['total_trades']} trades, {metrics['win_rate']:.1f}% win, "
                  f"{metrics['total_return_pct']:+.1f}% return")
        
        return {
            'symbol': symbol, 'timeframe': timeframe, 'strategy': strategy_name,
            'test_bars': len(test_features),
            'test_start': str(test_features.index[0]), 'test_end': str(test_features.index[-1]),
            'metrics': metrics, 'trades': trades,
        }
    
    def _calculate_metrics(self, trades: List, final_equity: float, max_dd: float) -> Dict:
        if not trades:
            return {'total_trades': 0, 'winning_trades': 0, 'win_rate': 0, 
                    'total_return_pct': 0, 'max_drawdown_pct': 0, 'total_pnl_dollars': 0}
        
        wins = [t for t in trades if t['pnl_pips'] > 0]
        total_pnl = sum(t['pnl_dollars'] for t in trades)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'win_rate': round(len(wins) / len(trades) * 100, 2),
            'total_pnl_dollars': round(total_pnl, 2),
            'total_return_pct': round((final_equity - self.config.initial_capital) / self.config.initial_capital * 100, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'final_equity': round(final_equity, 2),
        }
    
    def _generate_report(self, elapsed: float, verbose: bool) -> Dict:
        if not self.simulation_results:
            return {'status': 'No results'}
        
        total_trades = sum(r['metrics']['total_trades'] for r in self.simulation_results.values())
        total_wins = sum(r['metrics']['winning_trades'] for r in self.simulation_results.values())
        total_pnl = sum(r['metrics'].get('total_pnl_dollars', 0) for r in self.simulation_results.values())
        max_dd = max(r['metrics']['max_drawdown_pct'] for r in self.simulation_results.values())
        
        summary = {
            'process': 'Live Simulation (TEST DATA)',
            'symbols_simulated': len(self.simulation_results),
            'total_trades': total_trades,
            'win_rate': round(total_wins / total_trades * 100, 2) if total_trades > 0 else 0,
            'total_pnl_dollars': round(total_pnl, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'time_sec': round(elapsed, 2),
        }
        
        checks = {
            'has_trades': total_trades > 0,
            'positive_pnl': total_pnl > 0,
            'acceptable_drawdown': max_dd < 30,
        }
        summary['validation_checks'] = checks
        summary['all_passed'] = all(checks.values())
        
        if verbose:
            print("\n" + "=" * 70)
            print("📊 FINAL VALIDATION REPORT")
            print("=" * 70)
            print(f"\n{'Symbol':<10} {'Trades':>7} {'Win%':>7} {'Return%':>10} {'MaxDD%':>8}")
            print("-" * 45)
            
            for symbol, result in sorted(self.simulation_results.items(), 
                                         key=lambda x: x[1]['metrics']['total_return_pct'], reverse=True):
                m = result['metrics']
                print(f"{symbol:<10} {m['total_trades']:>7} {m['win_rate']:>7.1f} "
                      f"{m['total_return_pct']:>+10.1f} {m['max_drawdown_pct']:>8.1f}")
            
            print("\n" + "-" * 45)
            print(f"Total P&L: ${total_pnl:+,.2f}")
            print(f"Max Drawdown: {max_dd:.1f}%")
            
            print("\n" + "=" * 70)
            print("🎉 ALL CHECKS PASSED!" if summary['all_passed'] else "⚠️ SOME CHECKS FAILED")
            print("=" * 70)
        
        return summary
    
    def _export_results(self, summary: Dict) -> None:
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(to_native(summary), f, indent=2)
        
        if self.simulation_results:
            rows = []
            for symbol, result in self.simulation_results.items():
                m = result['metrics']
                rows.append({
                    'symbol': symbol, 'timeframe': result['timeframe'], 'strategy': result['strategy'],
                    'test_bars': result['test_bars'], 'total_trades': m['total_trades'],
                    'win_rate': m['win_rate'], 'total_return_pct': m['total_return_pct'],
                    'max_drawdown_pct': m['max_drawdown_pct'],
                })
            pd.DataFrame(rows).to_csv(self.output_dir / 'simulation_results.csv', index=False)
            
            all_trades = []
            for symbol, result in self.simulation_results.items():
                for trade in result['trades']:
                    trade['symbol'] = symbol
                    all_trades.append(trade)
            if all_trades:
                pd.DataFrame(all_trades).to_csv(self.output_dir / 'all_trades.csv', index=False)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ProductionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = DataLoader(config.data_dir)
        self.feature_computer = None
        self.data_splitter = DataSplitter(config.train_pct, config.val_pct, config.test_pct)
        self.strategy_registry = StrategyRegistry()
    
    def run_all(self, verbose: bool = True):
        print("=" * 70)
        print("FX PORTFOLIO MANAGER - PRODUCTION PIPELINE v2.1")
        print("=" * 70)
        print(f"\nData: {self.config.data_dir}")
        print(f"Output: {self.config.output_dir}")
        print(f"Split: {self.config.train_pct}% / {self.config.val_pct}% / {self.config.test_pct}%")
        print(f"Strategies: {len(self.strategy_registry.list_strategies())}")
        
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        self.data_loader.load_all_symbols()
        
        if not self.data_loader.raw_data:
            print("❌ No data loaded.")
            return
        
        self.feature_computer = FeatureComputer(self.data_loader)
        
        with Timer("Process 1"):
            Process1_StrategyApplication(
                self.config, self.data_loader, self.feature_computer,
                self.data_splitter, self.strategy_registry
            ).run(verbose)
        
        with Timer("Process 2"):
            Process2_StrategyEvaluation(
                self.config, self.data_loader, self.feature_computer,
                self.data_splitter, self.strategy_registry
            ).run(verbose)
        
        with Timer("Process 3"):
            Process3_Optimization(
                self.config, self.data_loader, self.feature_computer,
                self.data_splitter, self.strategy_registry
            ).run(verbose)
        
        with Timer("Process 4"):
            Process4_LiveSimulation(
                self.config, self.data_loader, self.feature_computer,
                self.data_splitter, self.strategy_registry
            ).run(verbose)
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutputs: {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='FX Portfolio Production Pipeline v2.1')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./fx_portfolio_outputs', help='Output directory')
    parser.add_argument('--train-pct', type=float, default=70.0)
    parser.add_argument('--val-pct', type=float, default=15.0)
    parser.add_argument('--test-pct', type=float, default=15.0)
    parser.add_argument('--capital', type=float, default=10000.0)
    parser.add_argument('--risk-pct', type=float, default=1.0)
    parser.add_argument('--min-trades', type=int, default=20)
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk_pct,
        min_trades=args.min_trades,
    )
    
    pipeline = ProductionPipeline(config)
    pipeline.run_all()


if __name__ == '__main__':
    main()
