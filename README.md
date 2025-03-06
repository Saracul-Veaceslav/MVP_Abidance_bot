# Abidance Crypto Trading Bot

A minimalist, yet powerful cryptocurrency trading bot designed for simplicity and performance.

## Features

- Connect to major exchanges (starting with Binance)
- Implement proven trading strategies
- Backtest strategies against historical data
- Basic risk management controls
- Command-line interface for easy operation
- Data persistence for trade history and configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/abidance.git
cd abidance
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root directory with your exchange API credentials:

```
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret
```

## Usage

### Basic Usage

```bash
abidance --exchange binance --symbol BTC/USDT --strategy ma_crossover
```

### Running Tests

```bash
pytest
```

## Project Structure

- `abidance/` - Main package
  - `exchange/` - Exchange integrations
  - `strategy/` - Trading strategies
  - `risk/` - Risk management
  - `data/` - Data persistence
  - `cli/` - Command-line interface
- `tests/` - Test suite

## Development Roadmap

1. Phase 1: Project Setup & Core Domain Models
2. Phase 2: Exchange Integration
3. Phase 3: Trading Strategies
4. Phase 4: Risk Management
5. Phase 5: Data Persistence
6. Phase 6: Command-Line Interface

## License

MIT 