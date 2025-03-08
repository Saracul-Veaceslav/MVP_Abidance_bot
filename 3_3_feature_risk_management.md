# 3.3 Risk Management

## User Story
As a trader, I want robust risk management controls so that the system can protect my capital and prevent catastrophic losses.

## Implementation Details

### 3.3.1 Position Sizing Controls

#### Position Sizer
```python
# risk/position_sizing.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class PositionSizeResult:
    size: float
    risk_amount: float
    risk_percentage: float
    max_loss: float
    reason: Optional[str] = None

class PositionSizer:
    def __init__(
        self,
        risk_per_trade: float = 0.01,  # 1% of account equity
        max_position_size: float = 0.25,  # 25% of account equity
        sizing_model: str = "fixed_risk"
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.sizing_model = sizing_model
        self.min_trade_size = 0.0001  # Minimum trade size
        
    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float = None,
        volatility: float = None,
        current_exposure: float = 0
    ) -> PositionSizeResult:
        """Calculate position size based on risk parameters."""
        try:
            # Calculate maximum position value
            max_position_value = account_equity * self.max_position_size
            
            # Calculate available position value considering current exposure
            available_value = max_position_value - current_exposure
            if available_value <= 0:
                return PositionSizeResult(
                    size=0,
                    risk_amount=0,
                    risk_percentage=0,
                    max_loss=0,
                    reason="Maximum position size reached"
                )
            
            if self.sizing_model == "fixed_risk" and stop_loss_price:
                # Calculate risk per unit
                risk_per_unit = abs(entry_price - stop_loss_price)
                if risk_per_unit == 0:
                    return PositionSizeResult(
                        size=0,
                        risk_amount=0,
                        risk_percentage=0,
                        max_loss=0,
                        reason="Invalid stop loss - same as entry"
                    )
                
                # Calculate risk amount in account currency
                risk_amount = account_equity * self.risk_per_trade
                
                # Calculate position size
                position_size = risk_amount / risk_per_unit
                
            elif self.sizing_model == "volatility_adjusted" and volatility:
                # Adjust risk based on volatility
                adjusted_risk = self.risk_per_trade * (1 / volatility)
                capped_risk = min(adjusted_risk, self.risk_per_trade * 2)
                
                risk_amount = account_equity * capped_risk
                position_size = risk_amount / entry_price
                
            else:
                # Default to fixed percentage of equity
                risk_amount = account_equity * self.risk_per_trade
                position_size = risk_amount / entry_price
            
            # Apply maximum position size limit
            max_size = available_value / entry_price
            position_size = min(position_size, max_size)
            
            # Apply minimum position size
            if position_size < self.min_trade_size:
                return PositionSizeResult(
                    size=0,
                    risk_amount=0,
                    risk_percentage=0,
                    max_loss=0,
                    reason="Position size below minimum"
                )
            
            # Calculate actual risk metrics
            actual_risk_amount = position_size * (entry_price - (stop_loss_price or entry_price * 0.99))
            actual_risk_percentage = actual_risk_amount / account_equity
            
            return PositionSizeResult(
                size=position_size,
                risk_amount=actual_risk_amount,
                risk_percentage=actual_risk_percentage,
                max_loss=position_size * entry_price
            )
            
        except Exception as e:
            return PositionSizeResult(
                size=0,
                risk_amount=0,
                risk_percentage=0,
                max_loss=0,
                reason=f"Error calculating position size: {str(e)}"
            )
```

### 3.3.2 Stop-Loss and Take-Profit Mechanisms

#### Stop Level Manager
```python
# risk/stop_loss.py
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class StopLevels:
    stop_loss: float
    take_profit: List[Tuple[float, float]]  # (price, size_percentage)
    trailing_activation: Optional[float] = None

class StopLossManager:
    def __init__(self, default_stop_type: str = "fixed"):
        self.default_stop_type = default_stop_type
        
    def calculate_stop_levels(
        self,
        entry_price: float,
        direction: str,  # 'long' or 'short'
        data: pd.DataFrame = None,
        stop_type: str = None,
        risk_reward_ratio: float = 2.0,
        **kwargs
    ) -> StopLevels:
        """Calculate stop loss and take profit levels."""
        stop_type = stop_type or self.default_stop_type
        
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(
            entry_price, direction, data, stop_type, **kwargs
        )
        
        # Calculate risk amount
        risk_amount = abs(entry_price - stop_loss)
        
        # Calculate take profit levels (multiple targets)
        take_profit_levels = []
        
        # First target at 1x risk/reward
        take_profit_levels.append((
            entry_price + (risk_amount * 1.0 * (1 if direction == 'long' else -1)),
            0.3  # Exit 30% of position
        ))
        
        # Second target at 2x risk/reward
        take_profit_levels.append((
            entry_price + (risk_amount * 2.0 * (1 if direction == 'long' else -1)),
            0.5  # Exit 50% of position
        ))
        
        # Final target at 3x risk/reward
        take_profit_levels.append((
            entry_price + (risk_amount * 3.0 * (1 if direction == 'long' else -1)),
            0.2  # Exit remaining 20%
        ))
        
        # Set trailing stop activation at 1.5x risk/reward
        trailing_activation = entry_price + (
            risk_amount * 1.5 * (1 if direction == 'long' else -1)
        )
        
        return StopLevels(
            stop_loss=stop_loss,
            take_profit=take_profit_levels,
            trailing_activation=trailing_activation
        )
        
    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        data: pd.DataFrame = None,
        stop_type: str = "fixed",
        **kwargs
    ) -> float:
        """Calculate stop loss price based on specified parameters."""
        if stop_type == "fixed":
            # Fixed percentage stop loss
            percentage = kwargs.get('percentage', 0.02)  # Default 2%
            if direction == 'long':
                stop_price = entry_price * (1 - percentage)
            else:
                stop_price = entry_price * (1 + percentage)
                
        elif stop_type == "atr":
            # ATR-based stop loss
            if data is None:
                raise ValueError("Data required for ATR-based stop loss")
            
            multiplier = kwargs.get('multiplier', 2.0)
            period = kwargs.get('period', 14)
            
            atr_value = atr(data, period).iloc[-1]
            
            if direction == 'long':
                stop_price = entry_price - (atr_value * multiplier)
            else:
                stop_price = entry_price + (atr_value * multiplier)
                
        elif stop_type == "recent_swing":
            # Recent swing high/low stop loss
            if data is None:
                raise ValueError("Data required for swing-based stop loss")
                
            lookback = kwargs.get('lookback', 10)
            
            if direction == 'long':
                # Stop below recent low
                stop_price = data['low'].rolling(lookback).min().iloc[-1]
            else:
                # Stop above recent high
                stop_price = data['high'].rolling(lookback).max().iloc[-1]
                
        else:
            raise ValueError(f"Unsupported stop type: {stop_type}")
            
        return stop_price
        
    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        direction: str,
        trail_type: str = "percentage",
        **kwargs
    ) -> float:
        """Update trailing stop based on current price."""
        if trail_type == "percentage":
            percentage = kwargs.get('percentage', 0.02)  # Default 2%
            
            if direction == 'long':
                new_stop = current_price * (1 - percentage)
                if new_stop > current_stop:
                    return new_stop
            else:
                new_stop = current_price * (1 + percentage)
                if new_stop < current_stop:
                    return new_stop
                    
        elif trail_type == "atr":
            atr_value = kwargs.get('atr_value')
            multiplier = kwargs.get('multiplier', 2.0)
            
            if direction == 'long':
                new_stop = current_price - (atr_value * multiplier)
                if new_stop > current_stop:
                    return new_stop
            else:
                new_stop = current_price + (atr_value * multiplier)
                if new_stop < current_stop:
                    return new_stop
                    
        return current_stop  # No change if new stop is not better
```

### 3.3.3 Maximum Drawdown Limits

#### Circuit Breaker
```python
# risk/circuit_breakers.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

@dataclass
class CircuitBreakerStatus:
    is_active: bool
    reason: Optional[str] = None
    reset_time: Optional[datetime] = None
    current_drawdown: float = 0.0
    daily_loss: float = 0.0

class CircuitBreaker:
    def __init__(
        self,
        daily_loss_limit: float = 0.03,  # 3% daily loss limit
        max_drawdown: float = 0.15,  # 15% max drawdown
        cooldown_period: int = 24,  # hours
        risk_reduction_thresholds: Dict[float, float] = None  # drawdown: risk_multiplier
    ):
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.cooldown_period = cooldown_period
        self.risk_reduction_thresholds = risk_reduction_thresholds or {
            0.05: 0.75,  # At 5% drawdown, reduce risk to 75%
            0.10: 0.50,  # At 10% drawdown, reduce risk to 50%
            0.12: 0.25   # At 12% drawdown, reduce risk to 25%
        }
        
        self.peak_equity = 0.0
        self.daily_starting_equity = 0.0
        self.triggered_time = None
        self.last_reset = datetime.now()
        
    def check_status(self, current_equity: float) -> CircuitBreakerStatus:
        """Check circuit breaker status and return current state."""
        now = datetime.now()
        
        # Initialize daily values if needed
        if self._is_new_trading_day(now):
            self.daily_starting_equity = current_equity
            self.last_reset = now
        
        # Update peak equity if we have a new high
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # Calculate drawdowns
        current_drawdown = 1 - (current_equity / self.peak_equity)
        daily_loss = 1 - (current_equity / self.daily_starting_equity)
        
        # Check if circuit breaker was previously triggered
        if self.triggered_time:
            cooldown_elapsed = now - self.triggered_time
            if cooldown_elapsed < timedelta(hours=self.cooldown_period):
                return CircuitBreakerStatus(
                    is_active=True,
                    reason="Cooling down from previous trigger",
                    reset_time=self.triggered_time + timedelta(hours=self.cooldown_period),
                    current_drawdown=current_drawdown,
                    daily_loss=daily_loss
                )
        
        # Check maximum drawdown
        if current_drawdown >= self.max_drawdown:
            self.triggered_time = now
            return CircuitBreakerStatus(
                is_active=True,
                reason=f"Maximum drawdown of {self.max_drawdown:.1%} exceeded",
                reset_time=now + timedelta(hours=self.cooldown_period),
                current_drawdown=current_drawdown,
                daily_loss=daily_loss
            )
            
        # Check daily loss limit
        if daily_loss >= self.daily_loss_limit:
            self.triggered_time = now
            return CircuitBreakerStatus(
                is_active=True,
                reason=f"Daily loss limit of {self.daily_loss_limit:.1%} exceeded",
                reset_time=now + timedelta(hours=self.cooldown_period),
                current_drawdown=current_drawdown,
                daily_loss=daily_loss
            )
            
        return CircuitBreakerStatus(
            is_active=False,
            current_drawdown=current_drawdown,
            daily_loss=daily_loss
        )
        
    def get_risk_multiplier(self, current_equity: float) -> float:
        """Get risk adjustment multiplier based on current drawdown."""
        if self.peak_equity == 0:
            return 1.0
            
        current_drawdown = 1 - (current_equity / self.peak_equity)
        
        # Find the appropriate risk reduction level
        for threshold, multiplier in sorted(
            self.risk_reduction_thresholds.items(),
            reverse=True
        ):
            if current_drawdown >= threshold:
                return multiplier
                
        return 1.0  # No risk reduction needed
        
    def _is_new_trading_day(self, current_time: datetime) -> bool:
        """Check if we're in a new trading day."""
        return (
            current_time.date() > self.last_reset.date() or
            (current_time - self.last_reset) > timedelta(hours=24)
        )
```

### 3.3.4 Portfolio Risk Management

#### Portfolio Manager
```python
# risk/portfolio.py
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PortfolioRisk:
    total_exposure: float
    margin_used: float
    risk_exposure: float
    position_weights: Dict[str, float]
    correlations: Dict[str, Dict[str, float]]
    warnings: List[str]

class PortfolioManager:
    def __init__(
        self,
        max_total_exposure: float = 2.0,  # 200% of equity
        max_single_exposure: float = 0.5,  # 50% of equity
        max_correlation_exposure: float = 0.8,  # 80% of equity in correlated pairs
        correlation_threshold: float = 0.7  # Pairs above this are considered correlated
    ):
        self.max_total_exposure = max_total_exposure
        self.max_single_exposure = max_single_exposure
        self.max_correlation_exposure = max_correlation_exposure
        self.correlation_threshold = correlation_threshold
        
    def analyze_portfolio(
        self,
        positions: Dict[str, float],  # symbol: position_value
        equity: float,
        correlations: Dict[str, Dict[str, float]]
    ) -> PortfolioRisk:
        """Analyze portfolio risk metrics."""
        warnings = []
        
        # Calculate total exposure
        total_exposure = sum(abs(pos) for pos in positions.values())
        total_exposure_ratio = total_exposure / equity
        
        if total_exposure_ratio > self.max_total_exposure:
            warnings.append(
                f"Total exposure ({total_exposure_ratio:.1%}) exceeds maximum "
                f"({self.max_total_exposure:.1%})"
            )
            
        # Calculate position weights
        position_weights = {
            symbol: abs(pos) / equity 
            for symbol, pos in positions.items()
        }
        
        # Check individual position sizes
        for symbol, weight in position_weights.items():
            if weight > self.max_single_exposure:
                warnings.append(
                    f"Position in {symbol} ({weight:.1%}) exceeds maximum single "
                    f"exposure ({self.max_single_exposure:.1%})"
                )
                
        # Analyze correlated exposure
        correlated_groups = self._find_correlated_groups(
            correlations,
            self.correlation_threshold
        )
        
        for group in correlated_groups:
            group_exposure = sum(
                abs(positions.get(symbol, 0))
                for symbol in group
            ) / equity
            
            if group_exposure > self.max_correlation_exposure:
                warnings.append(
                    f"Exposure to correlated group {group} ({group_exposure:.1%}) "
                    f"exceeds maximum ({self.max_correlation_exposure:.1%})"
                )
                
        # Calculate margin utilization (simplified)
        margin_used = total_exposure / 4  # Assuming 4x leverage available
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            margin_used=margin_used,
            risk_exposure=total_exposure_ratio,
            position_weights=position_weights,
            correlations=correlations,
            warnings=warnings
        )
        
    def _find_correlated_groups(
        self,
        correlations: Dict[str, Dict[str, float]],
        threshold: float
    ) -> List[List[str]]:
        """Find groups of correlated symbols."""
        groups = []
        processed = set()
        
        for symbol in correlations:
            if symbol in processed:
                continue
                
            # Find all symbols correlated with the current one
            correlated = {symbol}
            for other, corr in correlations[symbol].items():
                if abs(corr) >= threshold:
                    correlated.add(other)
                    
            if len(correlated) > 1:
                groups.append(list(correlated))
                processed.update(correlated)
                
        return groups
```

## Error Handling and Edge Cases

### Position Sizing
- Handle invalid input parameters
- Implement minimum and maximum size limits
- Account for exchange-specific size constraints

### Stop Loss Management
- Handle price gaps and slippage
- Implement failsafe mechanisms
- Support for multiple exit strategies

### Risk Monitoring
- Track real-time risk metrics
- Implement alert thresholds
- Handle market data delays

## Testing Strategy

### Unit Tests
```python
# tests/unit/risk/test_position_sizing.py
import pytest
from decimal import Decimal
from abidance.risk.position_sizing import PositionSizer

class TestPositionSizer:
    """
    Feature: Position Sizing
    """
    
    def setup_method(self):
        self.sizer = PositionSizer(
            risk_per_trade=0.01,
            max_position_size=0.25
        )
        
    def test_fixed_risk_sizing(self):
        """
        Scenario: Calculate position size with fixed risk
          Given an account equity of 10000
          And an entry price of 100
          And a stop loss price of 98
          When calculating position size
          Then the position should risk 1% of equity
        """
        result = self.sizer.calculate_position_size(
            account_equity=10000,
            entry_price=100,
            stop_loss_price=98
        )
        
        # Verify risk amount is 1% of equity
        assert result.risk_amount == pytest.approx(100, rel=1e-2)
        
        # Verify position size is correct
        expected_size = 100 / 2  # Risk amount / risk per unit
        assert result.size == pytest.approx(expected_size, rel=1e-2)
        
    def test_maximum_position_size(self):
        """
        Scenario: Position size limited by maximum exposure
          Given an account equity of 10000
          And a maximum position size of 25%
          When calculating a position that would exceed the limit
          Then the position size should be capped
        """
        result = self.sizer.calculate_position_size(
            account_equity=10000,
            entry_price=100,
            stop_loss_price=99,
            current_exposure=2000  # Already have 20% exposure
        )
        
        # Verify position is limited
        max_additional_exposure = 500  # 5% of equity remaining
        assert result.size * 100 <= max_additional_exposure
        
    def test_volatility_adjusted_sizing(self):
        """
        Scenario: Calculate position size with volatility adjustment
          Given an account equity of 10000
          And high volatility conditions
          When calculating position size
          Then the position size should be reduced
        """
        sizer = PositionSizer(
            risk_per_trade=0.01,
            sizing_model="volatility_adjusted"
        )
        
        result = sizer.calculate_position_size(
            account_equity=10000,
            entry_price=100,
            volatility=2.0  # High volatility
        )
        
        # Verify position size is reduced
        normal_size = 1.0  # Size without volatility adjustment
        assert result.size < normal_size

# tests/unit/risk/test_circuit_breaker.py
class TestCircuitBreaker:
    """
    Feature: Circuit Breaker Risk Management
    """
    
    def setup_method(self):
        self.circuit_breaker = CircuitBreaker(
            daily_loss_limit=0.03,
            max_drawdown=0.15
        )
        self.initial_equity = 10000
        self.circuit_breaker.peak_equity = self.initial_equity
        self.circuit_breaker.daily_starting_equity = self.initial_equity
        
    def test_daily_loss_limit(self):
        """
        Scenario: Daily loss limit breach
          Given an initial equity of 10000
          When the daily loss exceeds 3%
          Then the circuit breaker should activate
        """
        # Simulate 3.5% daily loss
        current_equity = self.initial_equity * 0.965
        
        status = self.circuit_breaker.check_status(current_equity)
        
        assert status.is_active
        assert "Daily loss limit" in status.reason
        assert status.daily_loss > 0.03
        
    def test_max_drawdown(self):
        """
        Scenario: Maximum drawdown breach
          Given a peak equity of 10000
          When the drawdown exceeds 15%
          Then the circuit breaker should activate
        """
        # Simulate 16% drawdown
        current_equity = self.initial_equity * 0.84
        
        status = self.circuit_breaker.check_status(current_equity)
        
        assert status.is_active
        assert "Maximum drawdown" in status.reason
        assert status.current_drawdown > 0.15
        
    def test_risk_reduction(self):
        """
        Scenario: Progressive risk reduction
          Given increasing drawdown levels
          When checking risk multipliers
          Then risk should be progressively reduced
        """
        # Test different drawdown levels
        equity_levels = [
            self.initial_equity * 0.96,  # 4% drawdown
            self.initial_equity * 0.92,  # 8% drawdown
            self.initial_equity * 0.87   # 13% drawdown
        ]
        
        multipliers = [
            self.circuit_breaker.get_risk_multiplier(equity)
            for equity in equity_levels
        ]
        
        # Verify risk reduction is progressive
        assert multipliers[0] > multipliers[1] > multipliers[2]
``` 