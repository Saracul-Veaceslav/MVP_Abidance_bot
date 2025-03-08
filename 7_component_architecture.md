# 7. Component Architecture

## 7.1 Core Components

### Data Processing Architecture

#### Market Data Collection
```python
# src/data/collectors/binance_collector.py
import asyncio
import logging
from typing import Dict, List, Optional

from binance import AsyncClient
from src.config import settings
from src.models.market_data import CandlestickData, OrderBookSnapshot

class BinanceDataCollector:
    """Collects real-time and historical market data from Binance."""
    
    def __init__(self, api_key: str = settings.BINANCE_API_KEY, 
                 api_secret: str = settings.BINANCE_API_SECRET):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the Binance client."""
        self.client = await AsyncClient.create(self.api_key, self.api_secret)
        self.logger.info("Binance client initialized")
        
    async def get_historical_klines(self, symbol: str, interval: str, 
                                   start_time: Optional[int] = None, 
                                   end_time: Optional[int] = None, 
                                   limit: int = 500) -> List[CandlestickData]:
        """
        Fetch historical candlestick data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Candlestick interval (e.g., '1m', '1h', '1d')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of candles to fetch
            
        Returns:
            List of CandlestickData objects
        """
        if not self.client:
            await self.initialize()
            
        try:
            klines = await self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time,
                limit=limit
            )
            
            return [
                CandlestickData(
                    symbol=symbol,
                    interval=interval,
                    open_time=int(kline[0]),
                    open_price=float(kline[1]),
                    high_price=float(kline[2]),
                    low_price=float(kline[3]),
                    close_price=float(kline[4]),
                    volume=float(kline[5]),
                    close_time=int(kline[6]),
                    quote_asset_volume=float(kline[7]),
                    number_of_trades=int(kline[8]),
                    taker_buy_base_volume=float(kline[9]),
                    taker_buy_quote_volume=float(kline[10])
                )
                for kline in klines
            ]
        except Exception as e:
            self.logger.error(f"Error fetching historical klines: {e}")
            raise
            
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBookSnapshot:
        """
        Fetch current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Depth of the order book
            
        Returns:
            OrderBookSnapshot object
        """
        if not self.client:
            await self.initialize()
            
        try:
            order_book = await self.client.get_order_book(symbol=symbol, limit=limit)
            
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=int(order_book['lastUpdateId']),
                bids=[(float(price), float(qty)) for price, qty in order_book['bids']],
                asks=[(float(price), float(qty)) for price, qty in order_book['asks']]
            )
        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            raise
            
    async def close(self):
        """Close the Binance client connection."""
        if self.client:
            await self.client.close_connection()
            self.logger.info("Binance client connection closed")

# src/data/collectors/websocket_manager.py
import asyncio
import json
import logging
from typing import Dict, List, Callable, Any

import websockets
from src.config import settings
from src.models.market_data import TradeUpdate, KlineUpdate, OrderBookUpdate

class BinanceWebsocketManager:
    """Manages WebSocket connections to Binance for real-time data."""
    
    def __init__(self):
        self.connections = {}
        self.callbacks = {}
        self.running = False
        self.logger = logging.getLogger(__name__)
        self.base_url = "wss://stream.binance.com:9443/ws"
        
    async def connect(self, streams: List[str], callback: Callable[[str, Any], None]):
        """
        Connect to Binance WebSocket streams.
        
        Args:
            streams: List of stream names to connect to
            callback: Function to call when data is received
        """
        stream_url = f"{self.base_url}/{'/'.join(streams)}"
        
        try:
            connection = await websockets.connect(stream_url)
            self.connections[stream_url] = connection
            
            for stream in streams:
                if stream not in self.callbacks:
                    self.callbacks[stream] = []
                self.callbacks[stream].append(callback)
                
            self.logger.info(f"Connected to streams: {streams}")
            
            if not self.running:
                self.running = True
                asyncio.create_task(self._listen(stream_url))
                
        except Exception as e:
            self.logger.error(f"Error connecting to WebSocket: {e}")
            raise
            
    async def _listen(self, stream_url: str):
        """Listen for messages on a WebSocket connection."""
        connection = self.connections.get(stream_url)
        
        if not connection:
            return
            
        try:
            while self.running and connection.open:
                message = await connection.recv()
                data = json.loads(message)
                
                stream = data.get('stream', '').split('@')[0]
                
                if stream in self.callbacks:
                    for callback in self.callbacks[stream]:
                        asyncio.create_task(callback(stream, data))
                        
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"WebSocket connection closed: {stream_url}")
            
        except Exception as e:
            self.logger.error(f"Error in WebSocket listener: {e}")
            
        finally:
            await self.disconnect(stream_url)
            
    async def disconnect(self, stream_url: str = None):
        """
        Disconnect from WebSocket streams.
        
        Args:
            stream_url: URL of the stream to disconnect from, or None to disconnect all
        """
        if stream_url:
            connection = self.connections.pop(stream_url, None)
            if connection and connection.open:
                await connection.close()
                self.logger.info(f"Disconnected from: {stream_url}")
        else:
            for url, connection in self.connections.items():
                if connection and connection.open:
                    await connection.close()
            self.connections = {}
            self.logger.info("Disconnected from all WebSocket streams")
            
        if not self.connections:
            self.running = False
```

#### Data Processing Pipeline
```python
# src/data/processors/data_processor.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from src.models.market_data import CandlestickData
from src.utils.validation import validate_dataframe

class DataProcessor:
    """Processes raw market data into clean, normalized formats for analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def candlesticks_to_dataframe(self, candles: List[CandlestickData]) -> pd.DataFrame:
        """
        Convert a list of CandlestickData objects to a pandas DataFrame.
        
        Args:
            candles: List of CandlestickData objects
            
        Returns:
            DataFrame with OHLCV data and additional metrics
        """
        if not candles:
            return pd.DataFrame()
            
        # Extract data into a dictionary
        data = {
            'timestamp': [c.open_time for c in candles],
            'open': [c.open_price for c in candles],
            'high': [c.high_price for c in candles],
            'low': [c.low_price for c in candles],
            'close': [c.close_price for c in candles],
            'volume': [c.volume for c in candles],
            'quote_volume': [c.quote_asset_volume for c in candles],
            'trades': [c.number_of_trades for c in candles],
            'taker_buy_base_volume': [c.taker_buy_base_volume for c in candles],
            'taker_buy_quote_volume': [c.taker_buy_quote_volume for c in candles]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime and set as index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Validate the DataFrame
        validate_dataframe(df, required_columns=['open', 'high', 'low', 'close', 'volume'])
        
        return df
        
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a DataFrame by handling missing values and outliers.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean.fillna(method='ffill', inplace=True)
        
        # Handle remaining NaNs (if any at the beginning)
        df_clean.fillna(method='bfill', inplace=True)
        
        # Detect and handle outliers (using IQR method for price columns)
        for col in ['open', 'high', 'low', 'close']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Log outliers but don't remove them, just cap them
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            if not outliers.empty:
                self.logger.warning(f"Found {len(outliers)} outliers in {col} column")
                
            # Cap outliers
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
        return df_clean
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df_indicators = df.copy()
        
        # Simple Moving Averages
        df_indicators['sma_7'] = df_indicators['close'].rolling(window=7).mean()
        df_indicators['sma_25'] = df_indicators['close'].rolling(window=25).mean()
        df_indicators['sma_99'] = df_indicators['close'].rolling(window=99).mean()
        
        # Exponential Moving Averages
        df_indicators['ema_7'] = df_indicators['close'].ewm(span=7, adjust=False).mean()
        df_indicators['ema_25'] = df_indicators['close'].ewm(span=25, adjust=False).mean()
        df_indicators['ema_99'] = df_indicators['close'].ewm(span=99, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df_indicators['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df_indicators['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df_indicators['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_indicators['close'].ewm(span=26, adjust=False).mean()
        df_indicators['macd'] = ema_12 - ema_26
        df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
        df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
        
        # Bollinger Bands
        df_indicators['bb_middle'] = df_indicators['close'].rolling(window=20).mean()
        std_dev = df_indicators['close'].rolling(window=20).std()
        df_indicators['bb_upper'] = df_indicators['bb_middle'] + (std_dev * 2)
        df_indicators['bb_lower'] = df_indicators['bb_middle'] - (std_dev * 2)
        
        # Average True Range (ATR)
        high_low = df_indicators['high'] - df_indicators['low']
        high_close = (df_indicators['high'] - df_indicators['close'].shift()).abs()
        low_close = (df_indicators['low'] - df_indicators['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_indicators['atr_14'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df_indicators['volume_sma_20'] = df_indicators['volume'].rolling(window=20).mean()
        df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_sma_20']
        
        return df_indicators
```

#### Error Handling
```python
# src/utils/error_handling.py
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

import aiohttp
import asyncio
from binance.exceptions import BinanceAPIException

T = TypeVar('T')

logger = logging.getLogger(__name__)

class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    pass

class APIRateLimitError(DataCollectionError):
    """Exception raised when API rate limits are hit."""
    pass

class ConnectionError(DataCollectionError):
    """Exception raised when connection issues occur."""
    pass

class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass

def retry_async(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (BinanceAPIException, aiohttp.ClientError, asyncio.TimeoutError)
) -> Callable:
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            delay = retry_delay
            
            while True:
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    retries += 1
                    
                    # Check if it's a rate limit error
                    if isinstance(e, BinanceAPIException) and e.code == -1003:
                        logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                        continue
                        
                    # If we've exceeded max retries, re-raise the exception
                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                        raise
                        
                    # Otherwise, wait and retry
                    logger.warning(f"Attempt {retries}/{max_retries} failed with error: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                    
        return wrapper
    return decorator

def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator to handle API errors and convert them to appropriate exceptions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
            
        except BinanceAPIException as e:
            if e.code == -1003:  # Rate limit exceeded
                logger.error(f"API rate limit exceeded: {e}")
                raise APIRateLimitError(f"Rate limit exceeded: {e}")
                
            elif e.code == -1021:  # Timestamp for this request is outside of the recvWindow
                logger.error(f"Timestamp error: {e}")
                raise DataCollectionError(f"Timestamp error: {e}")
                
            else:
                logger.error(f"Binance API error: {e}")
                raise DataCollectionError(f"API error: {e}")
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Connection error: {e}")
            raise ConnectionError(f"Connection error: {e}")
            
    return wrapper
```

#### Type Definitions
```python
# src/models/market_data.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

@dataclass
class CandlestickData:
    """Represents a single candlestick (OHLCV) data point."""
    symbol: str
    interval: str
    open_time: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    close_time: int
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float
    
    @property
    def datetime_open(self) -> datetime:
        """Convert open_time to datetime."""
        return datetime.fromtimestamp(self.open_time / 1000)
        
    @property
    def datetime_close(self) -> datetime:
        """Convert close_time to datetime."""
        return datetime.fromtimestamp(self.close_time / 1000)

@dataclass
class OrderBookSnapshot:
    """Represents a snapshot of the order book at a specific time."""
    symbol: str
    timestamp: int
    bids: List[Tuple[float, float]]  # List of (price, quantity) tuples
    asks: List[Tuple[float, float]]  # List of (price, quantity) tuples
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)
        
    @property
    def best_bid(self) -> Tuple[float, float]:
        """Get the best bid (highest price)."""
        return max(self.bids, key=lambda x: x[0]) if self.bids else (0.0, 0.0)
        
    @property
    def best_ask(self) -> Tuple[float, float]:
        """Get the best ask (lowest price)."""
        return min(self.asks, key=lambda x: x[0]) if self.asks else (0.0, 0.0)
        
    @property
    def spread(self) -> float:
        """Calculate the bid-ask spread."""
        if not self.bids or not self.asks:
            return 0.0
        return self.best_ask[0] - self.best_bid[0]
        
    @property
    def spread_percentage(self) -> float:
        """Calculate the bid-ask spread as a percentage."""
        if not self.bids or not self.asks or self.best_ask[0] == 0:
            return 0.0
        return (self.best_ask[0] - self.best_bid[0]) / self.best_ask[0] * 100

@dataclass
class TradeUpdate:
    """Represents a real-time trade update from WebSocket."""
    symbol: str
    trade_id: int
    price: float
    quantity: float
    buyer_order_id: int
    seller_order_id: int
    trade_time: int
    is_buyer_maker: bool
    
    @property
    def datetime(self) -> datetime:
        """Convert trade_time to datetime."""
        return datetime.fromtimestamp(self.trade_time / 1000)

@dataclass
class KlineUpdate:
    """Represents a real-time candlestick update from WebSocket."""
    symbol: str
    interval: str
    start_time: int
    end_time: int
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: float
    trade_count: int
    is_final: bool
    
    @property
    def datetime_start(self) -> datetime:
        """Convert start_time to datetime."""
        return datetime.fromtimestamp(self.start_time / 1000)
        
    @property
    def datetime_end(self) -> datetime:
        """Convert end_time to datetime."""
        return datetime.fromtimestamp(self.end_time / 1000)

@dataclass
class OrderBookUpdate:
    """Represents a real-time order book update from WebSocket."""
    symbol: str
    update_id: int
    event_time: int
    bids_to_update: List[Tuple[float, float]]
    asks_to_update: List[Tuple[float, float]]
    
    @property
    def datetime(self) -> datetime:
        """Convert event_time to datetime."""
        return datetime.fromtimestamp(self.event_time / 1000)
```

## 7.2 Client Components

### State Management

#### Local State
```typescript
// hooks/useLocalState.ts
import { useState, useCallback } from 'react'

export function useLocalState<T>(initialState: T) {
  const [state, setState] = useState<T>(initialState)
  
  const updateState = useCallback((newState: Partial<T>) => {
    setState(prev => ({ ...prev, ...newState }))
  }, [])
  
  return [state, updateState] as const
}

// components/StrategyForm.tsx
'use client'

import { useLocalState } from '@/hooks/useLocalState'

interface StrategyFormState {
  name: string
  type: string
  parameters: Record<string, any>
  isActive: boolean
}

export function StrategyForm() {
  const [state, updateState] = useLocalState<StrategyFormState>({
    name: '',
    type: 'trend_following',
    parameters: {},
    isActive: false
  })
  
  // Form handlers...
}
```

#### Server State
```typescript
// hooks/useServerState.ts
import { useState, useEffect } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'

interface ServerState<T> {
  data: T | null
  isLoading: boolean
  error: Error | null
}

export function useServerState<T>(
  initialData: T | null = null
): ServerState<T> {
  const [state, setState] = useState<ServerState<T>>({
    data: initialData,
    isLoading: !initialData,
    error: null
  })
  
  const socket = useWebSocket()
  
  useEffect(() => {
    if (!socket) return
    
    socket.on('state_update', (update: T) => {
      setState(prev => ({
        ...prev,
        data: update,
        isLoading: false
      }))
    })
    
    socket.on('error', (error: Error) => {
      setState(prev => ({
        ...prev,
        error,
        isLoading: false
      }))
    })
    
    return () => {
      socket.off('state_update')
      socket.off('error')
    }
  }, [socket])
  
  return state
}

// components/PositionMonitor.tsx
'use client'

import { useServerState } from '@/hooks/useServerState'
import type { Position } from '@/types'

export function PositionMonitor() {
  const { data: positions, isLoading, error } = useServerState<Position[]>([])
  
  if (isLoading) return <PositionSkeleton />
  if (error) return <ErrorMessage error={error} />
  if (!positions?.length) return <EmptyState />
  
  return (
    <div className="space-y-4">
      {positions.map(position => (
        <PositionCard key={position.id} position={position} />
      ))}
    </div>
  )
}
```

### Event Handlers

#### Form Handlers
```typescript
// components/StrategyForm.tsx
'use client'

import { useTransition } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { createStrategyAction } from '@/actions/strategies'
import type { StrategyFormData } from '@/types'

export function StrategyForm() {
  const [isPending, startTransition] = useTransition()
  
  const form = useForm<StrategyFormData>({
    resolver: zodResolver(strategySchema),
    defaultValues: {
      name: '',
      type: 'trend_following',
      parameters: {}
    }
  })
  
  const onSubmit = form.handleSubmit((data) => {
    startTransition(async () => {
      try {
        await createStrategyAction(data)
        form.reset()
      } catch (error) {
        console.error('Failed to create strategy:', error)
      }
    })
  })
  
  return (
    <form onSubmit={onSubmit} className="space-y-6">
      {/* Form fields */}
      <Button type="submit" isLoading={isPending}>
        Create Strategy
      </Button>
    </form>
  )
}
```

#### Interactive Handlers
```typescript
// components/PositionCard.tsx
'use client'

import { useState } from 'react'
import { closePositionAction } from '@/actions/positions'
import type { Position } from '@/types'

interface PositionCardProps {
  position: Position
}

export function PositionCard({ position }: PositionCardProps) {
  const [isClosing, setIsClosing] = useState(false)
  
  const handleClose = async () => {
    try {
      setIsClosing(true)
      await closePositionAction(position.id)
    } catch (error) {
      console.error('Failed to close position:', error)
    } finally {
      setIsClosing(false)
    }
  }
  
  return (
    <Card>
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-medium">{position.symbol}</h3>
          <p className="text-sm text-gray-500">
            {position.direction} @ {position.entryPrice}
          </p>
        </div>
        <Button
          variant="outline"
          onClick={handleClose}
          isLoading={isClosing}
        >
          Close Position
        </Button>
      </div>
    </Card>
  )
}
```

### UI Interactions

#### Animations
```typescript
// components/AnimatedNumber.tsx
'use client'

import { useSpring, animated } from '@react-spring/web'

interface AnimatedNumberProps {
  value: number
  prefix?: string
  suffix?: string
  decimals?: number
}

export function AnimatedNumber({
  value,
  prefix = '',
  suffix = '',
  decimals = 2
}: AnimatedNumberProps) {
  const spring = useSpring({
    from: { value: 0 },
    to: { value },
    config: {
      tension: 300,
      friction: 20
    }
  })
  
  return (
    <animated.span>
      {spring.value.to(val => 
        `${prefix}${val.toFixed(decimals)}${suffix}`
      )}
    </animated.span>
  )
}

// components/TransitionLayout.tsx
'use client'

import { motion, AnimatePresence } from 'framer-motion'

interface TransitionLayoutProps {
  children: React.ReactNode
}

export function TransitionLayout({ children }: TransitionLayoutProps) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.2 }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  )
}
```

#### Gestures
```typescript
// components/SwipeableCard.tsx
'use client'

import { motion, useAnimation, PanInfo } from 'framer-motion'

interface SwipeableCardProps {
  onSwipe: (direction: 'left' | 'right') => void
  children: React.ReactNode
}

export function SwipeableCard({ onSwipe, children }: SwipeableCardProps) {
  const controls = useAnimation()
  
  const handleDragEnd = (
    _: any,
    info: PanInfo
  ) => {
    const threshold = 100
    
    if (info.offset.x > threshold) {
      controls.start({ x: '100%' })
      onSwipe('right')
    } else if (info.offset.x < -threshold) {
      controls.start({ x: '-100%' })
      onSwipe('left')
    } else {
      controls.start({ x: 0 })
    }
  }
  
  return (
    <motion.div
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      onDragEnd={handleDragEnd}
      animate={controls}
      className="touch-none"
    >
      {children}
    </motion.div>
  )
}
```

### Props Interface

#### Component Props
```typescript
// types/props.ts
import type { ReactNode } from 'react'

// Layout Props
export interface LayoutProps {
  children: ReactNode
  className?: string
}

// Data Display Props
export interface DataDisplayProps<T> {
  data: T
  isLoading?: boolean
  error?: Error | null
}

// Form Props
export interface FormProps<T> {
  initialData?: Partial<T>
  onSubmit: (data: T) => Promise<void>
  isSubmitting?: boolean
  error?: Error | null
}

// Interactive Props
export interface InteractiveProps {
  isDisabled?: boolean
  isLoading?: boolean
  onClick?: () => void
  onHover?: () => void
  onFocus?: () => void
}

// Animation Props
export interface AnimationProps {
  initial?: object
  animate?: object
  exit?: object
  transition?: object
  variants?: object
}

// Component-Specific Props
export interface StrategyCardProps extends InteractiveProps {
  strategy: {
    id: string
    name: string
    type: string
    isActive: boolean
    performance: {
      totalReturn: number
      winRate: number
      sharpeRatio: number
    }
  }
  onActivate?: () => void
  onDeactivate?: () => void
  onEdit?: () => void
}

export interface PositionCardProps extends InteractiveProps {
  position: {
    id: string
    symbol: string
    direction: 'long' | 'short'
    entryPrice: number
    currentPrice: number
    size: number
    pnl: number
    pnlPercentage: number
  }
  onClose?: () => void
}

export interface TradeCardProps extends InteractiveProps {
  trade: {
    id: string
    symbol: string
    direction: 'long' | 'short'
    entryPrice: number
    exitPrice: number
    size: number
    pnl: number
    pnlPercentage: number
    entryTime: Date
    exitTime: Date
  }
}
``` 