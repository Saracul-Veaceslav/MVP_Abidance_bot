# 3. Feature Specification

## 3.1 Data Collection & Processing

### User Story
As a trader, I want the system to collect and process real-time market data from Binance so that strategies can make informed decisions based on current market conditions.

### Implementation Details

#### 3.1.1 Binance API Integration

#### API Key Management
```python
# exchange/binance/credentials.py
from cryptography.fernet import Fernet
from typing import Optional
import os

class CredentialManager:
    """Secure management of API credentials."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or os.getenv('CREDENTIAL_KEY')
        self.fernet = Fernet(self.encryption_key.encode())
        
    def encrypt_credentials(self, api_key: str, api_secret: str) -> tuple[bytes, bytes]:
        """Encrypt API credentials for storage."""
        encrypted_key = self.fernet.encrypt(api_key.encode())
        encrypted_secret = self.fernet.encrypt(api_secret.encode())
        return encrypted_key, encrypted_secret
        
    def decrypt_credentials(self, encrypted_key: bytes, encrypted_secret: bytes) -> tuple[str, str]:
        """Decrypt stored API credentials."""
        api_key = self.fernet.decrypt(encrypted_key).decode()
        api_secret = self.fernet.decrypt(encrypted_secret).decode()
        return api_key, api_secret
```

#### WebSocket Connection Management
```python
# exchange/binance/websocket.py
import asyncio
import json
import logging
from typing import Dict, Set, Optional
from websockets import connect
from .models import MarketUpdate

class BinanceWebSocket:
    """Manages WebSocket connections to Binance."""
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.subscriptions: Set[str] = set()
        self.connection = None
        self.is_connected = False
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self.max_reconnect_delay = 60.0
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> None:
        """Establish WebSocket connection with automatic reconnection."""
        while True:
            try:
                async with connect(self.url) as websocket:
                    self.connection = websocket
                    self.is_connected = True
                    self.reconnect_delay = 1.0  # Reset delay on successful connection
                    
                    # Resubscribe to channels
                    if self.subscriptions:
                        await self._subscribe(self.subscriptions)
                    
                    await self._handle_messages()
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {str(e)}")
                self.is_connected = False
                
                # Exponential backoff
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(
                    self.reconnect_delay * 2,
                    self.max_reconnect_delay
                )
                
    async def _handle_messages(self) -> None:
        """Process incoming WebSocket messages."""
        while True:
            try:
                message = await self.connection.recv()
                data = json.loads(message)
                
                # Convert to internal model
                update = MarketUpdate.from_binance(data)
                
                # Emit event
                await self.events.emit('market_update', update)
                
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                raise  # Re-raise to trigger reconnection
                
    async def subscribe(self, symbols: Set[str], channels: Set[str]) -> None:
        """Subscribe to market data streams."""
        streams = {
            f"{symbol.lower()}@{channel}"
            for symbol in symbols
            for channel in channels
        }
        
        if self.is_connected:
            await self._subscribe(streams)
            
        self.subscriptions.update(streams)
        
    async def _subscribe(self, streams: Set[str]) -> None:
        """Send subscription message."""
        message = {
            "method": "SUBSCRIBE",
            "params": list(streams),
            "id": 1
        }
        await self.connection.send(json.dumps(message))

#### 3.1.2 Market Data Collection

#### Candlestick Data
```python
# data/models.py
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

@dataclass
class Candlestick:
    """Represents OHLCV data for a time period."""
    
    symbol: str
    timeframe: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    num_trades: int
    taker_buy_volume: Decimal
    taker_buy_quote_volume: Decimal
    
    @classmethod
    def from_binance(cls, data: dict) -> 'Candlestick':
        """Create candlestick from Binance data format."""
        return cls(
            symbol=data['s'],
            timeframe=data['i'],
            timestamp=datetime.fromtimestamp(data['t'] / 1000),
            open_price=Decimal(str(data['o'])),
            high_price=Decimal(str(data['h'])),
            low_price=Decimal(str(data['l'])),
            close_price=Decimal(str(data['c'])),
            volume=Decimal(str(data['v'])),
            num_trades=int(data['n']),
            taker_buy_volume=Decimal(str(data['V'])),
            taker_buy_quote_volume=Decimal(str(data['Q']))
        )
```

#### Order Book Management
```python
# data/models.py
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Tuple

@dataclass
class OrderBook:
    """Represents the current state of the order book."""
    
    symbol: str
    last_update_id: int
    bids: List[Tuple[Decimal, Decimal]]  # price, quantity pairs
    asks: List[Tuple[Decimal, Decimal]]  # price, quantity pairs
    
    def update(self, data: dict) -> None:
        """Update order book with incremental changes."""
        # Process bid updates
        for bid in data['b']:
            price = Decimal(str(bid[0]))
            quantity = Decimal(str(bid[1]))
            
            if quantity == 0:
                self.bids = [x for x in self.bids if x[0] != price]
            else:
                # Insert maintaining price order (descending)
                self._update_level(self.bids, price, quantity, reverse=True)
                
        # Process ask updates
        for ask in data['a']:
            price = Decimal(str(ask[0]))
            quantity = Decimal(str(ask[1]))
            
            if quantity == 0:
                self.asks = [x for x in self.asks if x[0] != price]
            else:
                # Insert maintaining price order (ascending)
                self._update_level(self.asks, price, quantity)
                
    def _update_level(
        self,
        levels: List[Tuple[Decimal, Decimal]],
        price: Decimal,
        quantity: Decimal,
        reverse: bool = False
    ) -> None:
        """Update a price level while maintaining order."""
        for i, (p, q) in enumerate(levels):
            if p == price:
                levels[i] = (price, quantity)
                return
            if (not reverse and price < p) or (reverse and price > p):
                levels.insert(i, (price, quantity))
                return
        levels.append((price, quantity))

#### 3.1.3 Data Processing and Normalization

#### Data Cleaning
```python
# data/processor.py
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats

class DataProcessor:
    """Processes and normalizes market data."""
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean market data by handling missing values and anomalies."""
        # Handle missing values
        df = df.ffill().bfill()
        
        # Detect and handle anomalies using z-score
        for col in ['open', 'high', 'low', 'close']:
            z_scores = stats.zscore(df[col])
            abs_z_scores = np.abs(z_scores)
            outlier_mask = abs_z_scores > 3  # 3 sigma rule
            df.loc[outlier_mask, col] = df[col].median()
            
        return df
        
    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """Normalize data to a standard scale."""
        result = df.copy()
        
        if method == 'minmax':
            for col in columns:
                min_val = result[col].min()
                max_val = result[col].max()
                result[col] = (result[col] - min_val) / (max_val - min_val)
                
        elif method == 'zscore':
            for col in columns:
                mean = result[col].mean()
                std = result[col].std()
                result[col] = (result[col] - mean) / std
                
        return result
```

#### 3.1.4 Technical Indicators

#### Moving Averages
```python
# data/indicators/moving_averages.py
import pandas as pd
import numpy as np
from typing import Optional

def sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()

def ema(data: pd.Series, period: int, alpha: Optional[float] = None) -> pd.Series:
    """Calculate Exponential Moving Average."""
    if alpha is None:
        alpha = 2 / (period + 1)
    return data.ewm(alpha=alpha, adjust=False).mean()

def wma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return data.rolling(window=period).apply(
        lambda x: np.sum(weights * x) / weights.sum(), raw=True
    )
```

#### Momentum Indicators
```python
# data/indicators/momentum.py
import pandas as pd
import numpy as np
from typing import Tuple

def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram."""
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
```

#### 3.1.5 Data Storage

#### Time Series Database
```python
# data/database.py
from typing import List, Optional
from datetime import datetime
import asyncpg
from .models import Candlestick

class TimeSeriesDatabase:
    """Manages time series data storage."""
    
    def __init__(self, connection_params: dict):
        self.params = connection_params
        self.pool = None
        
    async def connect(self) -> None:
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(**self.params)
        
    async def save_candlestick(self, candle: Candlestick) -> None:
        """Save a candlestick to the database."""
        query = """
        INSERT INTO candlesticks (
            symbol, timeframe, timestamp, open, high, low, close,
            volume, num_trades, taker_buy_volume, taker_buy_quote_volume
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE
        SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            num_trades = EXCLUDED.num_trades,
            taker_buy_volume = EXCLUDED.taker_buy_volume,
            taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                candle.symbol,
                candle.timeframe,
                candle.timestamp,
                candle.open_price,
                candle.high_price,
                candle.low_price,
                candle.close_price,
                candle.volume,
                candle.num_trades,
                candle.taker_buy_volume,
                candle.taker_buy_quote_volume
            )
```

## Error Handling and Edge Cases

### Network Issues
- Implement exponential backoff for reconnection attempts
- Cache latest data for temporary outages
- Maintain data consistency during reconnection

### Data Quality
- Validate incoming data format and values
- Handle missing or delayed data points
- Detect and handle price anomalies

### Resource Management
- Implement connection pooling
- Handle memory constraints with streaming data
- Optimize database operations

## Testing Strategy

### Unit Tests
```python
# tests/unit/data/test_processor.py
import pytest
import pandas as pd
import numpy as np
from abidance.data.processor import DataProcessor

class TestDataProcessor:
    """
    Feature: Market Data Processing
    """
    
    def setup_method(self):
        self.processor = DataProcessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 200],  # 200 is an outlier
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
    def test_clean_data(self):
        """
        Scenario: Clean market data with missing values and outliers
          Given a DataFrame with missing values and outliers
          When cleaning the data
          Then missing values should be filled
          And outliers should be handled
        """
        cleaned = self.processor.clean_data(self.sample_data)
        
        # Check missing values are filled
        assert not cleaned['open'].isna().any()
        
        # Check outlier was handled
        assert cleaned['open'].iloc[4] < 150  # Outlier should be adjusted
        
    def test_normalize_data(self):
        """
        Scenario: Normalize market data to standard scale
          Given a DataFrame with price and volume data
          When normalizing the data
          Then values should be scaled to [0, 1] range
        """
        normalized = self.processor.normalize_data(
            self.sample_data,
            columns=['close', 'volume'],
            method='minmax'
        )
        
        # Check values are in [0, 1] range
        assert normalized['close'].between(0, 1).all()
        assert normalized['volume'].between(0, 1).all()
```

### Integration Tests
```python
# tests/integration/data/test_storage.py
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from abidance.data.models import Candlestick
from abidance.data.database import TimeSeriesDatabase

class TestTimeSeriesDatabase:
    """
    Feature: Time Series Data Storage
    """
    
    @pytest.fixture
    async def database(self):
        """Set up test database connection."""
        db = TimeSeriesDatabase({
            'host': 'localhost',
            'port': 5432,
            'database': 'test_abidance',
            'user': 'test_user',
            'password': 'test_password'
        })
        await db.connect()
        return db
        
    @pytest.fixture
    def sample_candle(self):
        """Create a sample candlestick."""
        return Candlestick(
            symbol='BTC-USDT',
            timeframe='1m',
            timestamp=datetime.utcnow(),
            open_price=Decimal('50000.00'),
            high_price=Decimal('50100.00'),
            low_price=Decimal('49900.00'),
            close_price=Decimal('50050.00'),
            volume=Decimal('10.5'),
            num_trades=100,
            taker_buy_volume=Decimal('5.25'),
            taker_buy_quote_volume=Decimal('262500.00')
        )
        
    async def test_save_and_retrieve_candlestick(
        self,
        database,
        sample_candle
    ):
        """
        Scenario: Save and retrieve candlestick data
          Given a database connection
          And a candlestick data point
          When saving the candlestick
          Then it should be retrievable from the database
          And the retrieved data should match the original
        """
        # Save candlestick
        await database.save_candlestick(sample_candle)
        
        # Retrieve candlestick
        retrieved = await database.get_candlestick(
            symbol=sample_candle.symbol,
            timeframe=sample_candle.timeframe,
            timestamp=sample_candle.timestamp
        )
        
        # Verify data
        assert retrieved is not None
        assert retrieved.symbol == sample_candle.symbol
        assert retrieved.open_price == sample_candle.open_price
        assert retrieved.close_price == sample_candle.close_price
        assert retrieved.volume == sample_candle.volume
``` 