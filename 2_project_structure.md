# 2. Project Structure

## 2.1 Directory Structure

The project follows a modular structure that separates concerns and promotes testability:

```
abidance/
├── __init__.py
├── cli/                       # Command-line interface
│   ├── __init__.py
│   └── commands.py            # CLI command implementations
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── settings.py            # Global configuration
│   └── validation.py          # Configuration validation
├── core/                      # Core domain logic
│   ├── __init__.py
│   ├── events.py              # Event system
│   └── service.py             # Service container
├── data/                      # Data management
│   ├── __init__.py
│   ├── database.py            # Database connection
│   ├── models.py              # Data models
│   ├── repository.py          # Data access layer
│   ├── storage.py             # Data storage
│   └── indicators/            # Technical indicators
│       ├── __init__.py
│       ├── moving_averages.py # Moving average indicators
│       ├── momentum.py        # Momentum indicators
│       └── volatility.py      # Volatility indicators
├── exchange/                  # Exchange integration
│   ├── __init__.py
│   ├── binance/
│   │   ├── __init__.py
│   │   ├── client.py          # Binance REST client
│   │   ├── websocket.py       # Binance WebSocket client
│   │   └── mapper.py          # Data mapping
│   ├── models.py              # Exchange models
│   └── interface.py           # Exchange interface
├── risk/                      # Risk management
│   ├── __init__.py
│   ├── position_sizing.py     # Position sizing logic
│   ├── stop_loss.py           # Stop-loss mechanisms
│   ├── circuit_breakers.py    # Circuit breaker implementation
│   └── portfolio.py           # Portfolio risk management
├── strategy/                  # Trading strategies
│   ├── __init__.py
│   ├── base.py                # Strategy base class
│   ├── trend_following/
│   │   ├── __init__.py
│   │   ├── ma_crossover.py    # Moving average crossover
│   │   └── breakout.py        # Breakout detection
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py        # Feature engineering
│   │   ├── models.py          # ML model definitions
│   │   ├── ppo.py             # PPO algorithm
│   │   └── dqn.py             # DQN algorithm
│   └── factory.py             # Strategy factory
├── execution/                 # Trade execution
│   ├── __init__.py
│   ├── order.py               # Order management
│   ├── position.py            # Position tracking
│   └── trader.py              # Trading logic
├── performance/               # Performance tracking
│   ├── __init__.py
│   ├── metrics.py             # Performance metrics
│   ├── logger.py              # Trade logging
│   └── visualization.py       # Data visualization
└── web/                       # Web interface
    ├── __init__.py
    ├── server.py              # Web server
    ├── api/                   # API endpoints
    │   ├── __init__.py
    │   ├── routes.py          # API routes
    │   └── schemas.py         # API schemas
    ├── static/                # Static assets
    │   ├── css/
    │   └── js/
    └── templates/             # HTML templates
        ├── dashboard.html
        ├── configuration.html
        └── performance.html

tests/                         # Test suite
├── __init__.py
├── conftest.py                # Test fixtures
├── unit/                      # Unit tests
│   ├── __init__.py
│   ├── data/
│   ├── exchange/
│   ├── risk/
│   └── strategy/
├── integration/               # Integration tests
│   ├── __init__.py
│   └── ...
└── e2e/                       # End-to-end tests
    ├── __init__.py
    └── ...

scripts/                       # Utility scripts
├── setup_dev.sh               # Development setup
└── backtest.py                # Backtesting script

docs/                          # Documentation
├── architecture.md            # Architecture documentation
├── api.md                     # API documentation
└── strategies.md              # Strategy documentation
```

## 2.2 Module Responsibilities

### 2.2.1 Core Components

| Module | Responsibility |
|--------|----------------|
| `cli/` | Provides command-line interface for configuring and running the trading bot |
| `config/` | Manages configuration loading, validation, and access throughout the application |
| `core/` | Contains core domain logic, event system, and service container |
| `data/` | Handles data processing, storage, and technical indicator calculation |
| `exchange/` | Manages communication with cryptocurrency exchanges |
| `risk/` | Implements risk management strategies and controls |
| `strategy/` | Contains trading strategy implementations and framework |
| `execution/` | Handles order execution and position management |
| `performance/` | Tracks and analyzes trading performance |
| `web/` | Provides web interface for monitoring and configuration |

### 2.2.2 Support Components

| Component | Responsibility |
|-----------|----------------|
| `tests/` | Contains all test cases organized by test type |
| `scripts/` | Utility scripts for development, deployment, and maintenance |
| `docs/` | Project documentation |

## 2.3 Key Design Patterns

### 2.3.1 Strategy Pattern
Used for implementing different trading strategies with a common interface.

```python
# strategy/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import pandas as pd

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.is_active = False
        self.required_indicators = []
        self.required_timeframes = []
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize strategy state and validate parameters."""
        pass
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on data."""
        pass
        
    def validate_signal(self, signal: int, data: pd.DataFrame, current_idx: int) -> Tuple[bool, float]:
        """Validate a trading signal and return confidence score."""
        return True, 1.0
```

### 2.3.2 Repository Pattern
Used for data access abstraction, separating business logic from data storage details.

```python
# data/repository.py
from typing import List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from .models import Candle

class MarketDataRepository(ABC):
    """Abstract repository for market data access."""
    
    @abstractmethod
    async def save_candle(self, candle: Candle) -> None:
        """Save a candlestick to the repository."""
        pass
    
    @abstractmethod
    async def get_candles(self, 
                          symbol: str, 
                          timeframe: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 100) -> List[Candle]:
        """Retrieve candlesticks from the repository."""
        pass
```

### 2.3.3 Factory Pattern
Used for creating strategy instances and other complex objects.

```python
# strategy/factory.py
from typing import Dict, Any, Type
from .base import Strategy

class StrategyFactory:
    """Factory for creating and managing trading strategies."""
    
    def __init__(self):
        self.registry = {}
        self.default_params = {}
        
    def register_strategy(self, name: str, strategy_class: Type[Strategy], 
                          default_params: Dict[str, Any] = None) -> None:
        """Register a new strategy."""
        self.registry[name] = strategy_class
        if default_params:
            self.default_params[name] = default_params
            
    def create_strategy(self, name: str, parameters: Dict[str, Any] = None) -> Strategy:
        """Create a new strategy instance."""
        if name not in self.registry:
            raise ValueError(f"Strategy {name} not found")
            
        strategy_class = self.registry[name]
        final_params = {**self.default_params.get(name, {}), **(parameters or {})}
        
        strategy = strategy_class(final_params)
        strategy.initialize()
        
        return strategy
```

### 2.3.4 Observer Pattern
Used for implementing the event system for component communication.

```python
# core/events.py
from typing import Dict, List, Any, Callable
from asyncio import Queue, create_task

class EventEmitter:
    """Event emitter for asynchronous event handling."""
    
    def __init__(self):
        self.listeners = {}
        self.queue = Queue()
        self._running = False
        
    def on(self, event_type: str, callback: Callable) -> None:
        """Register an event listener."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
            
        self.listeners[event_type].append(callback)
        
    def off(self, event_type: str, callback: Callable) -> None:
        """Remove an event listener."""
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
            
    async def emit(self, event_type: str, data: Any = None) -> None:
        """Emit an event."""
        await self.queue.put((event_type, data))
        
    async def start_processing(self) -> None:
        """Start processing events from the queue."""
        self._running = True
        
        while self._running:
            event_type, data = await self.queue.get()
            
            if event_type in self.listeners:
                for callback in self.listeners[event_type]:
                    create_task(callback(data))
            
            self.queue.task_done()
            
    async def stop_processing(self) -> None:
        """Stop processing events."""
        self._running = False
```

### 2.3.5 Dependency Injection
Used throughout the application to promote testability and loose coupling.

```python
# core/service.py
from typing import Dict, Any, Type

class ServiceContainer:
    """Container for dependency injection."""
    
    def __init__(self):
        self.services = {}
        self.factories = {}
        
    def register(self, service_name: str, instance: Any) -> None:
        """Register a service instance."""
        self.services[service_name] = instance
        
    def register_factory(self, service_name: str, factory) -> None:
        """Register a factory function for lazy initialization."""
        self.factories[service_name] = factory
        
    def get(self, service_name: str) -> Any:
        """Get a service by name, initializing if necessary."""
        if service_name in self.services:
            return self.services[service_name]
            
        if service_name in self.factories:
            service = self.factories[service_name](self)
            self.services[service_name] = service
            return service
            
        raise KeyError(f"Service {service_name} not found")
```

## 2.4 File Naming Conventions

- Python modules: Snake case (`moving_averages.py`)
- Classes: Pascal case (`class MovingAverageCrossover:`)
- Functions and variables: Snake case (`def calculate_rsi():`)
- Constants: Upper snake case (`MAX_RETRY_COUNT = 3`)

## 2.5 Import Organization

1. Standard library imports
2. Third-party imports
3. Application imports, organized by relativeness

Example:
```python
# Standard library
import os
import logging
from typing import List, Dict, Optional

# Third-party
import numpy as np
import pandas as pd
from fastapi import FastAPI

# Application
from abidance.core.events import EventEmitter
from abidance.data.repository import MarketDataRepository
```

## 2.6 Module Structure Standards

Each module should follow this general structure:

1. Module docstring explaining purpose
2. Imports (organized as above)
3. Constants
4. Classes and functions
5. Main executable code (if applicable)

Example:
```python
"""
Provides technical indicators based on moving averages.
"""

# Imports...

# Constants
DEFAULT_SMA_PERIOD = 20
DEFAULT_EMA_PERIOD = 12

# Functions and classes
def sma(data: pd.Series, period: int = DEFAULT_SMA_PERIOD) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()

# Main executable code (if applicable)
if __name__ == "__main__":
    # Example code
    pass
```

## 2.7 Configuration Management

The project uses a layered configuration approach:

1. **Default Configuration**: Hardcoded sensible defaults
2. **Environment Variables**: Override defaults with environment variables
3. **Configuration File**: Override with settings from YAML/JSON files
4. **Command-line Arguments**: Override with CLI arguments

```python
# config/settings.py
import os
import yaml
from typing import Dict, Any, Optional

class Settings:
    """Global configuration settings."""
    
    # Default configuration
    DEFAULTS = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "abidance",
            "user": "postgres",
            "password": "",
            "pool_size": 10
        },
        "binance": {
            "base_url": "https://api.binance.com",
            "ws_url": "wss://stream.binance.com:9443/ws",
            "request_timeout": 10.0,
            "max_retries": 3,
            "retry_delay": 1.0
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "abidance.log"
        },
        "trading": {
            "paper_trading": True,
            "max_open_positions": 5,
            "default_risk_per_trade": 0.01
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self.DEFAULTS.copy()
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                self._deep_update(self.config, file_config)
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Database settings
        if os.getenv("ABIDANCE_DB_HOST"):
            self.config["database"]["host"] = os.getenv("ABIDANCE_DB_HOST")
        # ... and so on for other env variables
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path."""
        parts = path.split('.')
        value = self.config
        
        for part in parts:
            if part not in value:
                return default
            value = value[part]
            
        return value
```

## 2.8 Initialization and Bootstrapping

The application follows a structured initialization process:

1. Load configuration
2. Initialize logging
3. Set up database connection
4. Register services and dependencies
5. Start event processing
6. Initialize subsystems

```python
# __main__.py
import asyncio
import logging
from abidance.config.settings import Settings
from abidance.core.service import ServiceContainer
from abidance.core.events import EventEmitter
from abidance.data.database import Database
from abidance.exchange.binance.client import BinanceClient

async def main():
    # Initialize configuration
    settings = Settings()
    
    # Set up logging
    logging_config = settings.get('logging')
    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format=logging_config['format'],
        filename=logging_config['file']
    )
    
    # Create service container
    container = ServiceContainer()
    
    # Register core services
    container.register('settings', settings)
    container.register('events', EventEmitter())
    
    # Initialize database
    db_config = settings.get('database')
    database = Database(
        host=db_config['host'],
        port=db_config['port'],
        name=db_config['name'],
        user=db_config['user'],
        password=db_config['password'],
        pool_size=db_config['pool_size']
    )
    await database.connect()
    container.register('database', database)
    
    # Initialize exchange client
    binance_config = settings.get('binance')
    binance_client = BinanceClient(
        api_key=os.getenv('BINANCE_API_KEY', ''),
        api_secret=os.getenv('BINANCE_API_SECRET', ''),
        base_url=binance_config['base_url'],
        ws_url=binance_config['ws_url'],
        timeout=binance_config['request_timeout'],
        max_retries=binance_config['max_retries'],
        retry_delay=binance_config['retry_delay'],
        events=container.get('events')
    )
    container.register('exchange_client', binance_client)
    
    # Start event processing
    events = container.get('events')
    asyncio.create_task(events.start_processing())
    
    # Initialize other subsystems...
    
    # Run forever
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Cleanup resources
        await events.stop_processing()
        await database.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 