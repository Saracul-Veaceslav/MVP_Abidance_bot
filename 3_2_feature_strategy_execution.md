# 3.2 Strategy Execution

## User Story
As a trader, I want the system to execute trading strategies based on market data and indicators so that it can automatically identify and act on profitable trading opportunities.

## Implementation Details

### 3.2.1 Strategy Base Framework

#### Strategy Interface
```python
# strategy/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, parameters: Dict[str, any] = None):
        self.parameters = parameters or {}
        self.is_active = False
        self.required_indicators = []
        self.required_timeframes = []
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize strategy state and validate parameters."""
        pass
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate strategy-specific indicators."""
        pass
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on indicator values."""
        pass
        
    def validate_signal(
        self, 
        signal: int, 
        data: pd.DataFrame, 
        current_idx: int
    ) -> Tuple[bool, float]:
        """Validate a trading signal and return confidence score."""
        return True, 1.0
        
    def optimize_timing(
        self, 
        signal: int, 
        data: pd.DataFrame, 
        current_idx: int
    ) -> int:
        """Optimize the timing of signal execution."""
        return current_idx
```

### 3.2.2 Trend Following Strategy Implementation

#### Moving Average Crossover Strategy
```python
# strategy/trend_following/ma_crossover.py
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from ..base import Strategy
from ...data.indicators.moving_averages import sma, ema
from ...data.indicators.volatility import average_true_range

class MACrossoverStrategy(Strategy):
    """Moving average crossover strategy."""
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        signal_period: int = 3,
        ma_type: str = "ema"
    ):
        super().__init__({
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "ma_type": ma_type
        })
        self.required_indicators = ["moving_averages"]
        
    def initialize(self) -> None:
        """Validate strategy parameters."""
        if self.parameters["fast_period"] >= self.parameters["slow_period"]:
            raise ValueError("Fast period must be less than slow period")
            
        if self.parameters["ma_type"] not in ["sma", "ema"]:
            raise ValueError("Unsupported MA type")
            
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate moving averages."""
        price = data['close']
        
        if self.parameters["ma_type"] == "sma":
            fast_ma = sma(price, self.parameters["fast_period"])
            slow_ma = sma(price, self.parameters["slow_period"])
        else:
            fast_ma = ema(price, self.parameters["fast_period"])
            slow_ma = ema(price, self.parameters["slow_period"])
            
        return {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma
        }
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossovers."""
        indicators = self.calculate_indicators(data)
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Calculate crossovers
        crossover = (indicators['fast_ma'] > indicators['slow_ma']).astype(int).diff()
        
        # Generate signals
        signals.loc[crossover == 1, 'signal'] = 1  # Buy signal
        signals.loc[crossover == -1, 'signal'] = -1  # Sell signal
        
        return signals
        
    def validate_signal(
        self, 
        signal: int, 
        data: pd.DataFrame, 
        current_idx: int
    ) -> Tuple[bool, float]:
        """
        Validate signals with additional confirmation rules.
        Returns (is_valid, confidence_score).
        """
        if current_idx < self.parameters["slow_period"]:
            return False, 0.0
            
        # Get recent data window
        window = data.iloc[current_idx - 20:current_idx + 1]
        
        # Calculate trend strength
        atr = average_true_range(window, period=14)
        trend_strength = abs(window['close'].diff(20).iloc[-1]) / (atr.iloc[-1] * 20)
        
        # Calculate volume confirmation
        volume_sma = window['volume'].rolling(20).mean()
        volume_confirmation = window['volume'].iloc[-1] > volume_sma.iloc[-1]
        
        # Combined confirmation
        is_valid = trend_strength > 0.5 and volume_confirmation
        confidence = min(1.0, trend_strength) * 0.8 + (0.2 if volume_confirmation else 0)
        
        return is_valid, confidence
```

#### Breakout Strategy
```python
# strategy/trend_following/breakout.py
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from ..base import Strategy
from ...data.indicators.volatility import average_true_range, bollinger_bands

class BreakoutStrategy(Strategy):
    """Price breakout detection strategy."""
    
    def __init__(
        self,
        lookback_period: int = 20,
        volatility_factor: float = 2.0,
        volume_factor: float = 1.5
    ):
        super().__init__({
            "lookback_period": lookback_period,
            "volatility_factor": volatility_factor,
            "volume_factor": volume_factor
        })
        self.required_indicators = ["volatility"]
        
    def initialize(self) -> None:
        """Validate strategy parameters."""
        if self.parameters["lookback_period"] < 10:
            raise ValueError("Lookback period must be at least 10")
            
        if self.parameters["volatility_factor"] <= 0:
            raise ValueError("Volatility factor must be positive")
            
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate breakout levels."""
        lookback = self.parameters["lookback_period"]
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(
            data['close'],
            period=lookback,
            std_dev=self.parameters["volatility_factor"]
        )
        
        # Calculate ATR for volatility-based stops
        atr = average_true_range(data, period=lookback)
        
        return {
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "atr": atr
        }
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on breakouts."""
        indicators = self.calculate_indicators(data)
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Detect breakouts
        upper_break = (data['close'] > indicators['bb_upper']) & \
                     (data['close'].shift(1) <= indicators['bb_upper'])
                     
        lower_break = (data['close'] < indicators['bb_lower']) & \
                     (data['close'].shift(1) >= indicators['bb_lower'])
        
        # Generate signals
        signals.loc[upper_break, 'signal'] = 1  # Bullish breakout
        signals.loc[lower_break, 'signal'] = -1  # Bearish breakout
        
        return signals
        
    def validate_signal(
        self, 
        signal: int, 
        data: pd.DataFrame, 
        current_idx: int
    ) -> Tuple[bool, float]:
        """Validate breakout signals."""
        if current_idx < self.parameters["lookback_period"]:
            return False, 0.0
            
        # Get recent data window
        window = data.iloc[current_idx - 20:current_idx + 1]
        
        # Volume confirmation
        volume_sma = window['volume'].rolling(20).mean()
        volume_surge = window['volume'].iloc[-1] > \
                      volume_sma.iloc[-1] * self.parameters["volume_factor"]
        
        # Momentum confirmation
        price_momentum = window['close'].pct_change(5).iloc[-1]
        momentum_aligned = (signal > 0 and price_momentum > 0) or \
                         (signal < 0 and price_momentum < 0)
        
        # Combined validation
        is_valid = volume_surge and momentum_aligned
        confidence = 0.7 if volume_surge else 0.0
        confidence += 0.3 if momentum_aligned else 0.0
        
        return is_valid, confidence
```

### 3.2.3 Machine Learning Strategy Framework

#### Feature Engineering Pipeline
```python
# strategy/ml/features.py
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ...data.indicators import moving_averages, momentum, volatility

class FeatureEngineer:
    """Prepares features for ML models."""
    
    def __init__(self, feature_config: Dict[str, Any]):
        self.config = feature_config
        self.scalers = {}
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix from raw market data."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        if self.config.get('use_price_features', True):
            features = self._add_price_features(data, features)
            
        # Technical indicators
        if self.config.get('use_technical_indicators', True):
            features = self._add_technical_indicators(data, features)
            
        # Volume features
        if self.config.get('use_volume_features', True):
            features = self._add_volume_features(data, features)
            
        # Time-based features
        if self.config.get('use_time_features', True):
            features = self._add_time_features(data, features)
            
        return features
        
    def _add_price_features(
        self, 
        data: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        for period in [1, 5, 10, 20]:
            features[f'return_{period}'] = data['close'].pct_change(period)
            
        # Price levels
        features['price_to_sma_20'] = data['close'] / \
            moving_averages.sma(data['close'], 20)
            
        features['price_to_sma_50'] = data['close'] / \
            moving_averages.sma(data['close'], 50)
            
        # Price patterns
        features['hl_range'] = (data['high'] - data['low']) / data['close']
        features['oc_range'] = abs(data['open'] - data['close']) / data['close']
        
        return features
        
    def _add_technical_indicators(
        self, 
        data: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add technical indicator features."""
        # Momentum indicators
        features['rsi_14'] = momentum.rsi(data['close'], 14)
        
        macd_line, signal_line, hist = momentum.macd(
            data['close'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_hist'] = hist
        
        # Volatility indicators
        bb_upper, bb_middle, bb_lower = volatility.bollinger_bands(
            data['close'],
            period=20,
            std_dev=2
        )
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        features['atr_14'] = volatility.average_true_range(data, 14)
        
        return features
        
    def _add_volume_features(
        self, 
        data: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume momentum
        features['volume_sma_ratio'] = data['volume'] / \
            moving_averages.sma(data['volume'], 20)
            
        # Volume trends
        for period in [5, 10, 20]:
            features[f'volume_change_{period}'] = \
                data['volume'].pct_change(period)
                
        # Price-volume relationship
        features['price_volume_corr'] = data['close'].rolling(20).corr(data['volume'])
        
        return features
        
    def _add_time_features(
        self, 
        data: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add time-based features."""
        timestamp = pd.to_datetime(data.index)
        
        # Cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
        
        features['day_sin'] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        
        return features
        
    def normalize_features(
        self, 
        features: pd.DataFrame, 
        fit: bool = False
    ) -> pd.DataFrame:
        """Normalize features using StandardScaler."""
        if fit:
            self.scalers = {}
            for column in features.columns:
                scaler = StandardScaler()
                features[column] = scaler.fit_transform(
                    features[[column]]
                ).flatten()
                self.scalers[column] = scaler
        else:
            for column, scaler in self.scalers.items():
                if column in features.columns:
                    features[column] = scaler.transform(
                        features[[column]]
                    ).flatten()
                    
        return features
```

#### Model Interface
```python
# strategy/ml/models.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLModel(ABC):
    """Base class for ML trading models."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.model = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture."""
        pass
        
    @abstractmethod
    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """Train the model."""
        pass
        
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass
        
    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'parameters': self.parameters
            }, path)
            
    def load(self, path: str) -> None:
        """Load model from disk."""
        if not self.model:
            self.build_model()
            
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.parameters = checkpoint['parameters']
```

### 3.2.4 Strategy Factory and Registry

#### Strategy Factory
```python
# strategy/factory.py
from typing import Dict, Type, Any
from .base import Strategy
from .trend_following import MACrossoverStrategy, BreakoutStrategy
from .ml import MLStrategy

class StrategyFactory:
    """Factory for creating and managing trading strategies."""
    
    def __init__(self):
        self.registry = {}
        self.default_params = {}
        
    def register_strategy(
        self, 
        name: str, 
        strategy_class: Type[Strategy],
        default_params: Dict[str, Any] = None
    ) -> None:
        """Register a new strategy."""
        if name in self.registry:
            raise ValueError(f"Strategy {name} already registered")
            
        self.registry[name] = strategy_class
        if default_params:
            self.default_params[name] = default_params
            
    def create_strategy(
        self, 
        name: str, 
        parameters: Dict[str, Any] = None
    ) -> Strategy:
        """Create a new strategy instance."""
        if name not in self.registry:
            raise ValueError(f"Strategy {name} not found")
            
        strategy_class = self.registry[name]
        
        # Merge default and provided parameters
        final_params = self.default_params.get(name, {}).copy()
        if parameters:
            final_params.update(parameters)
            
        # Create and initialize strategy
        strategy = strategy_class(final_params)
        strategy.initialize()
        
        return strategy
```

## Error Handling and Edge Cases

### Strategy Validation
- Validate strategy parameters before initialization
- Handle missing or invalid indicator data
- Implement signal confirmation rules

### Market Conditions
- Detect unfavorable market conditions
- Implement circuit breakers for extreme volatility
- Handle trading hour restrictions

### Resource Management
- Optimize memory usage for feature calculation
- Cache frequently used indicators
- Implement efficient data structures

## Testing Strategy

### Unit Tests
```python
# tests/unit/strategy/test_ma_crossover.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abidance.strategy.trend_following import MACrossoverStrategy

class TestMACrossoverStrategy:
    """
    Feature: Moving Average Crossover Strategy
    """
    
    def setup_method(self):
        self.strategy = MACrossoverStrategy(
            fast_period=10,
            slow_period=20,
            ma_type="ema"
        )
        
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(102, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
    def test_initialization(self):
        """
        Scenario: Strategy initialization with invalid parameters
          Given a strategy with fast period >= slow period
          When initializing the strategy
          Then a ValueError should be raised
        """
        with pytest.raises(ValueError):
            strategy = MACrossoverStrategy(
                fast_period=20,
                slow_period=20
            )
            strategy.initialize()
            
    def test_signal_generation(self):
        """
        Scenario: Generating trading signals
          Given price data with clear trend changes
          When generating signals
          Then buy signals should be generated on upward crossovers
          And sell signals should be generated on downward crossovers
        """
        # Initialize strategy
        self.strategy.initialize()
        
        # Generate signals
        signals = self.strategy.generate_signals(self.sample_data)
        
        # Verify signal properties
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert set(signals['signal'].unique()).issubset({-1, 0, 1})
        
    def test_signal_validation(self):
        """
        Scenario: Validating trading signals
          Given a valid trading signal
          When validating the signal
          Then confirmation rules should be applied
          And a confidence score should be returned
        """
        # Initialize strategy
        self.strategy.initialize()
        
        # Generate and validate a signal
        signals = self.strategy.generate_signals(self.sample_data)
        signal_idx = signals[signals['signal'] != 0].index[0]
        
        is_valid, confidence = self.strategy.validate_signal(
            signals.loc[signal_idx, 'signal'],
            self.sample_data,
            self.sample_data.index.get_loc(signal_idx)
        )
        
        # Verify validation results
        assert isinstance(is_valid, bool)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
```

### Integration Tests
```python
# tests/integration/strategy/test_strategy_execution.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abidance.strategy import StrategyFactory
from abidance.data import MarketDataRepository
from abidance.execution import TradeExecutor

class TestStrategyExecution:
    """
    Feature: Strategy Execution Pipeline
    """
    
    @pytest.fixture
    async def setup(self):
        """Set up test environment."""
        # Create components
        factory = StrategyFactory()
        repository = MarketDataRepository()
        executor = TradeExecutor()
        
        # Register test strategy
        factory.register_strategy(
            "ma_crossover",
            MACrossoverStrategy,
            {"fast_period": 10, "slow_period": 20}
        )
        
        return factory, repository, executor
        
    async def test_end_to_end_execution(self, setup):
        """
        Scenario: End-to-end strategy execution
          Given a registered strategy
          And historical market data
          When executing the strategy
          Then signals should be generated
          And trades should be executed according to the signals
        """
        factory, repository, executor = setup
        
        # Create strategy
        strategy = factory.create_strategy("ma_crossover")
        
        # Get market data
        data = await repository.get_candles(
            symbol="BTC-USDT",
            timeframe="1h",
            limit=100
        )
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Execute trades
        for idx, row in signals.iterrows():
            if row['signal'] != 0:
                is_valid, confidence = strategy.validate_signal(
                    row['signal'],
                    data,
                    data.index.get_loc(idx)
                )
                
                if is_valid:
                    trade = await executor.execute_trade(
                        symbol="BTC-USDT",
                        side="buy" if row['signal'] > 0 else "sell",
                        confidence=confidence
                    )
                    
                    assert trade is not None
                    assert trade.status == "filled"
``` 