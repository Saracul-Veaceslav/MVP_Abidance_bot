# 3.4 Performance Monitoring and Reporting

## User Story
As a trader, I want comprehensive performance monitoring and reporting capabilities so that I can track the effectiveness of my trading strategies, analyze risk-adjusted returns, and make data-driven decisions for strategy optimization.

## Implementation Details

### 3.4.1 Trade Performance Tracking

#### Trade Metrics
```python
# performance/trade_metrics.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class TradeMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: float
    risk_adjusted_return: float

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.equity_curve = pd.Series()
        self.daily_returns = pd.Series()
        
    def add_trade(self, trade: Dict) -> None:
        """Add a completed trade to the performance tracker."""
        self.trades.append({
            'entry_time': trade['entry_time'],
            'exit_time': trade['exit_time'],
            'symbol': trade['symbol'],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'size': trade['size'],
            'pnl': trade['pnl'],
            'strategy': trade.get('strategy', 'unknown')
        })
        
        # Update equity curve
        self._update_equity_curve(trade)
        
    def calculate_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> TradeMetrics:
        """Calculate performance metrics for the specified period."""
        # Filter trades by time period if specified
        trades_df = pd.DataFrame(self.trades)
        if start_time:
            trades_df = trades_df[trades_df['entry_time'] >= start_time]
        if end_time:
            trades_df = trades_df[trades_df['exit_time'] <= end_time]
            
        if len(trades_df) == 0:
            return self._create_empty_metrics()
            
        # Calculate basic metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_trades = len(trades_df)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average trade metrics
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Calculate trade duration metrics
        trades_df['duration'] = (
            trades_df['exit_time'] - trades_df['entry_time']
        ).dt.total_seconds() / 3600  # Convert to hours
        
        avg_duration = trades_df['duration'].mean()
        
        # Calculate risk-adjusted metrics
        daily_returns = self.daily_returns[
            (self.daily_returns.index >= start_time if start_time else True) &
            (self.daily_returns.index <= end_time if end_time else True)
        ]
        
        risk_free_rate = 0.02  # Annualized risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        excess_returns = daily_returns - daily_rf
        sharpe = self._calculate_sharpe_ratio(excess_returns)
        sortino = self._calculate_sortino_ratio(excess_returns)
        
        # Calculate drawdown metrics
        max_dd, max_dd_duration = self._calculate_drawdown_metrics(
            self.equity_curve[
                (self.equity_curve.index >= start_time if start_time else True) &
                (self.equity_curve.index <= end_time if end_time else True)
            ]
        )
        
        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            largest_loss=losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            avg_trade_duration=avg_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            risk_adjusted_return=self._calculate_risk_adjusted_return(
                win_rate, profit_factor, max_dd
            )
        )
        
    def _update_equity_curve(self, trade: Dict) -> None:
        """Update the equity curve with a new trade."""
        date = pd.Timestamp(trade['exit_time']).date()
        
        if date not in self.equity_curve.index:
            # Initialize with previous day's value or 0
            prev_equity = (
                self.equity_curve.iloc[-1] 
                if len(self.equity_curve) > 0 
                else 0
            )
            self.equity_curve[date] = prev_equity
            
        self.equity_curve[date] += trade['pnl']
        
        # Calculate daily returns
        self.daily_returns = self.equity_curve.pct_change().fillna(0)
        
    def _calculate_sharpe_ratio(self, excess_returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(excess_returns) < 2:
            return 0.0
            
        return (
            np.sqrt(252) * 
            excess_returns.mean() / 
            excess_returns.std()
        )
        
    def _calculate_sortino_ratio(self, excess_returns: pd.Series) -> float:
        """Calculate annualized Sortino ratio."""
        if len(excess_returns) < 2:
            return 0.0
            
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
            
        return (
            np.sqrt(252) * 
            excess_returns.mean() / 
            downside_returns.std()
        )
        
    def _calculate_drawdown_metrics(
        self,
        equity_curve: pd.Series
    ) -> Tuple[float, float]:
        """Calculate maximum drawdown and drawdown duration."""
        if len(equity_curve) < 2:
            return 0.0, 0.0
            
        # Calculate drawdown series
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        # Find maximum drawdown
        max_drawdown = abs(drawdowns.min())
        
        # Calculate drawdown duration
        is_drawdown = drawdowns < 0
        drawdown_start = is_drawdown.astype(int).diff()
        
        current_duration = 0
        max_duration = 0
        
        for idx, value in drawdown_start.items():
            if value == 1:  # Start of drawdown
                current_duration = 0
            elif value == 0 and is_drawdown[idx]:  # Continuing drawdown
                current_duration += 1
                max_duration = max(max_duration, current_duration)
                
        return max_drawdown, max_duration
        
    def _calculate_risk_adjusted_return(
        self,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float
    ) -> float:
        """Calculate custom risk-adjusted return metric."""
        if max_drawdown == 0:
            return 0.0
            
        return (win_rate * profit_factor) / max_drawdown
        
    def _create_empty_metrics(self) -> TradeMetrics:
        """Create empty metrics when no trades are available."""
        return TradeMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_trade_duration=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0.0,
            risk_adjusted_return=0.0
        )
```

### 3.4.2 Real-time Monitoring

#### Position Monitor
```python
# performance/monitoring.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

@dataclass
class PositionStatus:
    symbol: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    duration: float
    risk_metrics: Dict[str, float]

@dataclass
class StrategyStatus:
    name: str
    is_active: bool
    positions: List[str]
    daily_signals: int
    daily_trades: int
    daily_pnl: float
    allocation: float
    risk_status: str

class MonitoringSystem:
    def __init__(self):
        self.positions = {}
        self.strategies = {}
        self.last_update = datetime.now()
        
    def update_position(
        self,
        symbol: str,
        current_price: float,
        risk_metrics: Optional[Dict] = None
    ) -> None:
        """Update position status with current market data."""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        position.current_price = current_price
        
        # Update unrealized P&L
        price_change = current_price - position.entry_price
        if position.direction == 'short':
            price_change = -price_change
            
        position.unrealized_pnl = price_change * position.size
        position.unrealized_pnl_pct = (
            position.unrealized_pnl / 
            (position.entry_price * position.size)
        )
        
        # Update duration
        position.duration = (
            datetime.now() - position.entry_time
        ).total_seconds() / 3600
        
        # Update risk metrics
        if risk_metrics:
            position.risk_metrics.update(risk_metrics)
            
    def update_strategy(
        self,
        name: str,
        signals: int,
        trades: int,
        pnl: float,
        allocation: float
    ) -> None:
        """Update strategy status with current performance data."""
        if name not in self.strategies:
            self.strategies[name] = StrategyStatus(
                name=name,
                is_active=True,
                positions=[],
                daily_signals=0,
                daily_trades=0,
                daily_pnl=0.0,
                allocation=allocation,
                risk_status='normal'
            )
            
        strategy = self.strategies[name]
        
        # Update daily metrics
        strategy.daily_signals = signals
        strategy.daily_trades = trades
        strategy.daily_pnl = pnl
        strategy.allocation = allocation
        
        # Update risk status based on performance
        self._update_strategy_risk_status(strategy)
        
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all open positions."""
        if not self.positions:
            return pd.DataFrame()
            
        return pd.DataFrame([
            {
                'symbol': symbol,
                'direction': pos.direction,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'duration': pos.duration,
                **pos.risk_metrics
            }
            for symbol, pos in self.positions.items()
        ])
        
    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary of all active strategies."""
        if not self.strategies:
            return pd.DataFrame()
            
        return pd.DataFrame([
            {
                'name': name,
                'is_active': strat.is_active,
                'position_count': len(strat.positions),
                'daily_signals': strat.daily_signals,
                'daily_trades': strat.daily_trades,
                'daily_pnl': strat.daily_pnl,
                'allocation': strat.allocation,
                'risk_status': strat.risk_status
            }
            for name, strat in self.strategies.items()
        ])
        
    def _update_strategy_risk_status(self, strategy: StrategyStatus) -> None:
        """Update strategy risk status based on performance metrics."""
        # Define risk thresholds
        DAILY_LOSS_THRESHOLD = -0.02  # 2% daily loss
        HIGH_ACTIVITY_THRESHOLD = 20   # trades per day
        
        if strategy.daily_pnl / strategy.allocation <= DAILY_LOSS_THRESHOLD:
            strategy.risk_status = 'high_risk'
        elif strategy.daily_trades >= HIGH_ACTIVITY_THRESHOLD:
            strategy.risk_status = 'high_activity'
        else:
            strategy.risk_status = 'normal'
```

### 3.4.3 Performance Analytics

#### Return Analysis
```python
# performance/analytics.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

@dataclass
class ReturnMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    best_month: float
    worst_month: float
    monthly_returns: pd.Series

@dataclass
class RiskMetrics:
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    treynor_ratio: float
    active_return: float
    active_risk: float

class PerformanceAnalytics:
    def __init__(
        self,
        equity_curve: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ):
        self.equity_curve = equity_curve
        self.returns = equity_curve.pct_change().fillna(0)
        self.benchmark_returns = benchmark_returns
        
    def calculate_return_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ReturnMetrics:
        """Calculate comprehensive return metrics."""
        # Filter data by date range
        returns = self._filter_date_range(self.returns, start_date, end_date)
        
        if len(returns) < 2:
            return self._create_empty_return_metrics()
            
        # Calculate basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate distribution metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Calculate Value at Risk and Conditional VaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calculate monthly metrics
        monthly_returns = self._calculate_monthly_returns(returns)
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        
        return ReturnMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            best_month=best_month,
            worst_month=worst_month,
            monthly_returns=monthly_returns
        )
        
    def calculate_risk_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        if self.benchmark_returns is None:
            return self._create_empty_risk_metrics()
            
        # Filter data by date range
        returns = self._filter_date_range(self.returns, start_date, end_date)
        bench_returns = self._filter_date_range(
            self.benchmark_returns, start_date, end_date
        )
        
        if len(returns) < 2 or len(bench_returns) < 2:
            return self._create_empty_risk_metrics()
            
        # Calculate beta and correlation
        covariance = returns.cov(bench_returns)
        variance = bench_returns.var()
        beta = covariance / variance if variance != 0 else 0
        correlation = returns.corr(bench_returns)
        
        # Calculate tracking error and information ratio
        tracking_diff = returns - bench_returns
        tracking_error = tracking_diff.std() * np.sqrt(252)
        
        active_return = returns.mean() - bench_returns.mean()
        information_ratio = (
            active_return * np.sqrt(252) / tracking_error 
            if tracking_error != 0 else 0
        )
        
        # Calculate Treynor ratio
        risk_free_rate = 0.02  # Annualized risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_return = returns.mean() - daily_rf
        treynor_ratio = (
            excess_return * 252 / beta 
            if beta != 0 else 0
        )
        
        # Calculate active metrics
        active_return = (
            (1 + returns).prod() - (1 + bench_returns).prod()
        )
        active_risk = tracking_error
        
        return RiskMetrics(
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            active_return=active_return,
            active_risk=active_risk
        )
        
    def _calculate_monthly_returns(
        self,
        returns: pd.Series
    ) -> pd.Series:
        """Calculate monthly return series."""
        return (
            (1 + returns).resample('M').prod() - 1
        ).fillna(0)
        
    def _filter_date_range(
        self,
        data: pd.Series,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.Series:
        """Filter data by date range."""
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data
        
    def _create_empty_return_metrics(self) -> ReturnMetrics:
        """Create empty return metrics."""
        return ReturnMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            skewness=0.0,
            kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
            best_month=0.0,
            worst_month=0.0,
            monthly_returns=pd.Series()
        )
        
    def _create_empty_risk_metrics(self) -> RiskMetrics:
        """Create empty risk metrics."""
        return RiskMetrics(
            beta=0.0,
            correlation=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0,
            active_return=0.0,
            active_risk=0.0
        )
```

## Error Handling and Edge Cases

### Data Gaps
- Handle missing data points
- Implement data validation
- Support for irregular time series

### Market Conditions
- Detect market regime changes
- Handle extreme volatility periods
- Monitor liquidity conditions

### Performance Anomalies
- Detect outlier returns
- Track unusual patterns
- Monitor strategy drift

## Testing Strategy

### Unit Tests
```python
# tests/unit/performance/test_trade_metrics.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abidance.performance.trade_metrics import PerformanceTracker

class TestPerformanceTracker:
    """
    Feature: Trade Performance Tracking
    """
    
    def setup_method(self):
        self.tracker = PerformanceTracker()
        self.sample_trades = [
            {
                'entry_time': datetime(2024, 1, 1, 10, 0),
                'exit_time': datetime(2024, 1, 1, 14, 0),
                'symbol': 'BTC-USD',
                'direction': 'long',
                'entry_price': 100.0,
                'exit_price': 105.0,
                'size': 1.0,
                'pnl': 5.0,
                'strategy': 'trend_following'
            },
            {
                'entry_time': datetime(2024, 1, 1, 15, 0),
                'exit_time': datetime(2024, 1, 1, 16, 0),
                'symbol': 'ETH-USD',
                'direction': 'short',
                'entry_price': 200.0,
                'exit_price': 195.0,
                'size': 0.5,
                'pnl': 2.5,
                'strategy': 'mean_reversion'
            }
        ]
        
    def test_trade_metrics_calculation(self):
        """
        Scenario: Calculate basic trade metrics
          Given a series of completed trades
          When calculating performance metrics
          Then the metrics should reflect the trading results
        """
        # Add sample trades
        for trade in self.sample_trades:
            self.tracker.add_trade(trade)
            
        metrics = self.tracker.calculate_metrics()
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 2
        assert metrics.win_rate == 1.0
        assert metrics.profit_factor > 0
        
    def test_risk_adjusted_metrics(self):
        """
        Scenario: Calculate risk-adjusted performance metrics
          Given a series of trades with varying profitability
          When calculating risk-adjusted metrics
          Then the metrics should account for both returns and risk
        """
        # Add trades with mixed results
        winning_trade = self.sample_trades[0]
        losing_trade = {
            **self.sample_trades[1],
            'exit_price': 205.0,
            'pnl': -2.5
        }
        
        self.tracker.add_trade(winning_trade)
        self.tracker.add_trade(losing_trade)
        
        metrics = self.tracker.calculate_metrics()
        
        assert metrics.sharpe_ratio is not None
        assert metrics.sortino_ratio is not None
        assert metrics.max_drawdown >= 0
        assert metrics.risk_adjusted_return is not None

# tests/unit/performance/test_monitoring.py
class TestMonitoringSystem:
    """
    Feature: Real-time Performance Monitoring
    """
    
    def setup_method(self):
        self.monitor = MonitoringSystem()
        self.sample_position = PositionStatus(
            symbol='BTC-USD',
            direction='long',
            size=1.0,
            entry_price=100.0,
            current_price=100.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            duration=0.0,
            risk_metrics={'leverage': 1.0}
        )
        
    def test_position_update(self):
        """
        Scenario: Update position status with market data
          Given an open position
          When the market price changes
          Then the position metrics should be updated
        """
        # Add position and update price
        self.monitor.positions['BTC-USD'] = self.sample_position
        self.monitor.update_position('BTC-USD', 105.0)
        
        position = self.monitor.positions['BTC-USD']
        assert position.current_price == 105.0
        assert position.unrealized_pnl == 5.0
        assert position.unrealized_pnl_pct == pytest.approx(0.05)
        
    def test_strategy_monitoring(self):
        """
        Scenario: Monitor strategy performance
          Given an active trading strategy
          When performance metrics are updated
          Then the strategy status should reflect current performance
        """
        self.monitor.update_strategy(
            name='trend_following',
            signals=10,
            trades=5,
            pnl=100.0,
            allocation=0.5
        )
        
        strategy = self.monitor.strategies['trend_following']
        assert strategy.is_active
        assert strategy.daily_signals == 10
        assert strategy.daily_trades == 5
        assert strategy.daily_pnl == 100.0
        assert strategy.allocation == 0.5
``` 