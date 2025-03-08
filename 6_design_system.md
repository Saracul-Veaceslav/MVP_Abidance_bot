# 6. Design System

## 6.1 Visual Style

### Color Palette
```python
# abidance/web/static/css/theme.py
from enum import Enum

class ColorScheme(Enum):
    """Color scheme for the trading bot UI."""
    # Primary Colors
    PRIMARY = {
        "50": "#F0F7FF",
        "100": "#E0EFFF",
        "200": "#B8DBFF", 
        "300": "#8FC7FF",
        "400": "#66B3FF",
        "500": "#3D9FFF",  # Main primary color
        "600": "#0A85FF",
        "700": "#006BE0",
        "800": "#0054B3",
        "900": "#003D80"
    }
    
    # Secondary Colors
    SECONDARY = {
        "50": "#F5F3FF",
        "100": "#EDE9FE",
        "200": "#DDD6FE",
        "300": "#C4B5FD",
        "400": "#A78BFA",
        "500": "#8B5CF6",  # Main secondary color
        "600": "#7C3AED",
        "700": "#6D28D9",
        "800": "#5B21B6",
        "900": "#4C1D95"
    }
    
    # Semantic Colors
    SUCCESS = "#059669"
    WARNING = "#D97706"
    ERROR = "#DC2626"
    INFO = "#2563EB"
    
    # Chart Colors
    PROFIT = "#10B981"  # Green
    LOSS = "#EF4444"    # Red
    NEUTRAL = "#6B7280" # Gray
    
    # Trading Signals
    BUY = "#10B981"     # Green
    SELL = "#EF4444"    # Red
    HOLD = "#6B7280"    # Gray

# Neutral Colors
NEUTRAL_COLORS = {
    "white": "#FFFFFF",
    "black": "#000000",
    "gray": {
        "50": "#F9FAFB",
        "100": "#F3F4F6",
        "200": "#E5E7EB",
        "300": "#D1D5DB",
        "400": "#9CA3AF",
        "500": "#6B7280",
        "600": "#4B5563",
        "700": "#374151",
        "800": "#1F2937",
        "900": "#111827"
  }
}
```

### Typography

```python
# abidance/web/static/css/typography.py

class Typography:
    """Typography settings for the trading bot UI."""
    FONT_FAMILY = {
        "main": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "mono": "'JetBrains Mono', 'Courier New', monospace",
        "data": "'IBM Plex Mono', 'Consolas', monospace"
    }
    
    FONT_SIZE = {
        "xs": "0.75rem",     # 12px
        "sm": "0.875rem",    # 14px
        "base": "1rem",      # 16px
        "lg": "1.125rem",    # 18px
        "xl": "1.25rem",     # 20px
        "2xl": "1.5rem",     # 24px
        "3xl": "1.875rem",   # 30px
        "4xl": "2.25rem",    # 36px
        "5xl": "3rem"        # 48px
    }
    
    FONT_WEIGHT = {
        "light": 300,
        "normal": 400,
        "medium": 500,
        "semibold": 600,
        "bold": 700
    }
    
    LINE_HEIGHT = {
        "none": 1,
        "tight": 1.25,
        "snug": 1.375,
        "normal": 1.5,
        "relaxed": 1.625,
        "loose": 2
    }
```

### Web Interface Theme

```python
# abidance/web/static/css/theme_config.py
from abidance.web.static.css.theme import ColorScheme, NEUTRAL_COLORS
from abidance.web.static.css.typography import Typography

class WebInterfaceTheme:
    """Theme configuration for the web interface."""
    # Component-specific themes
    DASHBOARD_CARD = {
        "background": NEUTRAL_COLORS["white"],
        "border": NEUTRAL_COLORS["gray"]["200"],
        "border_radius": "0.375rem",
        "shadow": "0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px 0 rgba(0,0,0,0.06)",
        "padding": "1.5rem",
        "header_font_size": Typography.FONT_SIZE["lg"],
        "header_font_weight": Typography.FONT_WEIGHT["medium"],
        "header_color": NEUTRAL_COLORS["gray"]["900"]
    }
    
    CHART = {
        "background": NEUTRAL_COLORS["white"],
        "grid_color": NEUTRAL_COLORS["gray"]["200"],
        "axis_color": NEUTRAL_COLORS["gray"]["500"],
        "label_color": NEUTRAL_COLORS["gray"]["700"],
        "label_font_size": Typography.FONT_SIZE["sm"],
        "tooltip_background": NEUTRAL_COLORS["gray"]["800"],
        "tooltip_text_color": NEUTRAL_COLORS["white"],
        "price_line_color": ColorScheme.PRIMARY["500"].value,
        "volume_bar_color": ColorScheme.PRIMARY["300"].value,
        "line_width": 1.5
    }
    
    TRADES_TABLE = {
        "header_background": NEUTRAL_COLORS["gray"]["100"],
        "header_text_color": NEUTRAL_COLORS["gray"]["700"],
        "row_border_color": NEUTRAL_COLORS["gray"]["200"],
        "row_hover_background": NEUTRAL_COLORS["gray"]["50"],
        "profit_color": ColorScheme.PROFIT.value,
        "loss_color": ColorScheme.LOSS.value,
        "buy_color": ColorScheme.BUY.value,
        "sell_color": ColorScheme.SELL.value,
        "font_family": Typography.FONT_FAMILY["data"]
    }
    
    PERFORMANCE_METRICS = {
        "positive_value_color": ColorScheme.PROFIT.value,
        "negative_value_color": ColorScheme.LOSS.value,
        "neutral_value_color": ColorScheme.NEUTRAL.value,
        "label_color": NEUTRAL_COLORS["gray"]["600"],
        "value_font_size": Typography.FONT_SIZE["xl"],
        "value_font_weight": Typography.FONT_WEIGHT["semibold"],
        "label_font_size": Typography.FONT_SIZE["sm"]
    }
    
    FORM_CONTROLS = {
        "input_border": NEUTRAL_COLORS["gray"]["300"],
        "input_background": NEUTRAL_COLORS["white"],
        "input_text": NEUTRAL_COLORS["gray"]["900"],
        "input_placeholder": NEUTRAL_COLORS["gray"]["400"],
        "input_focus_border": ColorScheme.PRIMARY["500"].value,
        "input_focus_ring": ColorScheme.PRIMARY["100"].value,
        "input_border_radius": "0.375rem",
        "input_padding": "0.5rem 0.75rem",
        "input_font_size": Typography.FONT_SIZE["base"],
        "label_color": NEUTRAL_COLORS["gray"]["700"],
        "label_font_size": Typography.FONT_SIZE["sm"],
        "label_font_weight": Typography.FONT_WEIGHT["medium"]
    }
    
    BUTTONS = {
        "primary": {
            "background": ColorScheme.PRIMARY["600"].value,
            "text_color": NEUTRAL_COLORS["white"],
            "hover_background": ColorScheme.PRIMARY["700"].value,
            "active_background": ColorScheme.PRIMARY["800"].value,
            "disabled_background": ColorScheme.PRIMARY["300"].value
        },
        "secondary": {
            "background": NEUTRAL_COLORS["white"],
            "text_color": NEUTRAL_COLORS["gray"]["700"],
            "border_color": NEUTRAL_COLORS["gray"]["300"],
            "hover_background": NEUTRAL_COLORS["gray"]["50"],
            "active_background": NEUTRAL_COLORS["gray"]["100"],
            "disabled_background": NEUTRAL_COLORS["gray"]["50"]
        },
        "danger": {
            "background": ColorScheme.ERROR.value,
            "text_color": NEUTRAL_COLORS["white"],
            "hover_background": "#B91C1C",  # Darker red
            "active_background": "#991B1B",  # Even darker red
            "disabled_background": "#FCA5A5"  # Lighter red
  }
}
```

## 6.2 Data Visualization

### Chart Styles

```python
# abidance/performance/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from abidance.web.static.css.theme import ColorScheme, NEUTRAL_COLORS

class ChartStyles:
    """Visualization styles and utilities for trading bot charts."""
    
    @staticmethod
    def set_style():
        """Set the default style for all charts."""
        plt.style.use('fivethirtyeight')
        sns.set(font_scale=1.1)
        
        # Customize plot appearance
        plt.rcParams['figure.facecolor'] = NEUTRAL_COLORS["white"]
        plt.rcParams['axes.facecolor'] = NEUTRAL_COLORS["white"]
        plt.rcParams['axes.edgecolor'] = NEUTRAL_COLORS["gray"]["300"]
        plt.rcParams['axes.labelcolor'] = NEUTRAL_COLORS["gray"]["800"]
        plt.rcParams['xtick.color'] = NEUTRAL_COLORS["gray"]["700"]
        plt.rcParams['ytick.color'] = NEUTRAL_COLORS["gray"]["700"]
        plt.rcParams['grid.color'] = NEUTRAL_COLORS["gray"]["200"]
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7
    
    @staticmethod
    def create_price_chart(df, ax=None, volume=True, signals=None):
        """
        Create a price chart with optional volume and signals.
        
        Args:
            df: DataFrame with 'timestamp', 'open', 'high', 'low', 'close', and optionally 'volume'
            ax: matplotlib axis to plot on (if None, one will be created)
            volume: Whether to include volume bars
            signals: DataFrame with buy/sell signals
            
        Returns:
            matplotlib axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot price line
        ax.plot(df['timestamp'], df['close'], 
                color=ColorScheme.PRIMARY["600"].value, 
                linewidth=1.5, 
                label='Close Price')
        
        # Add volume if requested
        if volume and 'volume' in df.columns:
            volume_ax = ax.twinx()
            volume_ax.bar(df['timestamp'], df['volume'], 
                         color=ColorScheme.PRIMARY["200"].value, 
                         alpha=0.3, 
                         width=0.8)
            volume_ax.set_ylabel('Volume')
            volume_ax.grid(False)
            volume_ax.spines['right'].set_color(NEUTRAL_COLORS["gray"]["300"])
            volume_ax.tick_params(axis='y', colors=NEUTRAL_COLORS["gray"]["500"])
        
        # Add buy/sell signals if provided
        if signals is not None:
            buy_signals = signals[signals['signal'] == 'buy']
            sell_signals = signals[signals['signal'] == 'sell']
            
            ax.scatter(buy_signals['timestamp'], buy_signals['price'], 
                      marker='^', color=ColorScheme.BUY.value, s=100, 
                      label='Buy Signal')
            ax.scatter(sell_signals['timestamp'], sell_signals['price'], 
                      marker='v', color=ColorScheme.SELL.value, s=100, 
                      label='Sell Signal')
        
        # Style the chart
        ax.set_title('Price Chart', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        return ax
    
    @staticmethod
    def create_performance_chart(equity_curve, benchmark=None, ax=None):
        """
        Create a performance chart comparing strategy equity curve to benchmark.
        
        Args:
            equity_curve: Series with equity curve values
            benchmark: Optional benchmark series for comparison
            ax: matplotlib axis to plot on
            
        Returns:
            matplotlib axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot equity curve
        ax.plot(equity_curve.index, equity_curve.values, 
                color=ColorScheme.PRIMARY["600"].value, 
                linewidth=2, 
                label='Strategy')
        
        # Add benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values, 
                   color=NEUTRAL_COLORS["gray"]["500"], 
                   linewidth=1.5, 
                   linestyle='--', 
                   label='Benchmark')
        
        # Style the chart
        ax.set_title('Equity Curve', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        return ax
    
    @staticmethod
    def create_drawdown_chart(drawdown_series, ax=None):
        """
        Create a drawdown chart.
        
        Args:
            drawdown_series: Series with drawdown values
            ax: matplotlib axis to plot on
            
        Returns:
            matplotlib axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.fill_between(drawdown_series.index, 0, drawdown_series.values * 100, 
                       color=ColorScheme.ERROR.value, alpha=0.3)
        ax.plot(drawdown_series.index, drawdown_series.values * 100, 
               color=ColorScheme.ERROR.value, linewidth=1)
        
        # Style the chart
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        return ax
    
    @staticmethod
    def create_heatmap(performance_matrix, ax=None):
        """
        Create a heatmap for parameter optimization results.
        
        Args:
            performance_matrix: DataFrame with performance results
            ax: matplotlib axis to plot on
            
        Returns:
            matplotlib axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(performance_matrix, annot=True, fmt=".2f", 
                   cmap="RdYlGn", center=0, ax=ax)
        
        # Style the chart
        ax.set_title('Parameter Optimization Results', fontweight='bold')
        
        return ax
```

### Dashboard Components

```python
# abidance/web/templates/components.py
from jinja2 import Template
from abidance.web.static.css.theme import ColorScheme, NEUTRAL_COLORS
from abidance.web.static.css.typography import Typography

class DashboardComponents:
    """HTML components for the dashboard UI."""
    
    @staticmethod
    def metric_card_template():
        """Template for a metric card component."""
        return Template("""
        <div class="metric-card">
            <div class="metric-card-header">
                <h3 class="metric-card-title">{{ title }}</h3>
                {% if tooltip %}
                <div class="metric-card-tooltip" title="{{ tooltip }}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                </div>
                {% endif %}
      </div>
            <div class="metric-card-value {{ value_class }}">{{ value }}</div>
            {% if change is not none %}
            <div class="metric-card-change {{ 'positive' if change >= 0 else 'negative' }}">
                <span class="change-arrow">{{ '↑' if change >= 0 else '↓' }}</span>
                <span class="change-value">{{ change|abs }}%</span>
    </div>
            {% endif %}
        </div>
        """)
    
    @staticmethod
    def position_card_template():
        """Template for a position card component."""
        return Template("""
        <div class="position-card">
            <div class="position-card-header">
                <h3 class="position-card-title">{{ symbol }}</h3>
                <span class="position-card-direction {{ direction }}">{{ direction }}</span>
            </div>
            <div class="position-card-details">
                <div class="position-detail">
                    <span class="detail-label">Entry Price</span>
                    <span class="detail-value">{{ entry_price }}</span>
                </div>
                <div class="position-detail">
                    <span class="detail-label">Current Price</span>
                    <span class="detail-value">{{ current_price }}</span>
                </div>
                <div class="position-detail">
                    <span class="detail-label">Size</span>
                    <span class="detail-value">{{ size }}</span>
                </div>
                <div class="position-detail">
                    <span class="detail-label">P&L</span>
                    <span class="detail-value {{ 'positive' if pnl >= 0 else 'negative' }}">
                        {{ pnl }} ({{ pnl_percentage }}%)
                    </span>
                </div>
            </div>
            <div class="position-card-actions">
                <button class="btn btn-danger" data-position-id="{{ id }}" data-action="close">
                    Close Position
                </button>
            </div>
        </div>
        """)
    
    @staticmethod
    def trade_row_template():
        """Template for a trade history row."""
        return Template("""
        <tr class="trade-row">
            <td class="trade-timestamp">{{ timestamp }}</td>
            <td class="trade-symbol">{{ symbol }}</td>
            <td class="trade-direction {{ direction }}">{{ direction }}</td>
            <td class="trade-entry-price">{{ entry_price }}</td>
            <td class="trade-exit-price">{{ exit_price }}</td>
            <td class="trade-size">{{ size }}</td>
            <td class="trade-pnl {{ 'positive' if pnl >= 0 else 'negative' }}">
                {{ pnl }} ({{ pnl_percentage }}%)
            </td>
        </tr>
        """)
    
    @staticmethod
    def strategy_card_template():
        """Template for a strategy configuration card."""
        return Template("""
        <div class="strategy-card">
            <div class="strategy-card-header">
                <h3 class="strategy-card-title">{{ name }}</h3>
                <div class="strategy-card-status">
                    <span class="status-indicator {{ 'active' if is_active else 'inactive' }}"></span>
                    <span class="status-text">{{ 'Active' if is_active else 'Inactive' }}</span>
                </div>
            </div>
            <div class="strategy-card-description">
                {{ description }}
            </div>
            <div class="strategy-card-parameters">
                <h4 class="parameters-title">Parameters</h4>
                <div class="parameters-list">
                    {% for param_name, param_value in parameters.items() %}
                    <div class="parameter-item">
                        <span class="parameter-name">{{ param_name }}</span>
                        <span class="parameter-value">{{ param_value }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
            <div class="strategy-card-actions">
                <button class="btn {{ 'btn-danger' if is_active else 'btn-primary' }}" 
                        data-strategy-id="{{ id }}" 
                        data-action="{{ 'deactivate' if is_active else 'activate' }}">
                    {{ 'Deactivate' if is_active else 'Activate' }}
                </button>
                <button class="btn btn-secondary" 
                        data-strategy-id="{{ id }}" 
                        data-action="edit">
                    Edit
                </button>
            </div>
        </div>
        """)
```

## 6.3 Terminal Interface

```python
# abidance/cli/ui.py
import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from abidance.web.static.css.theme import ColorScheme

console = Console()

class TerminalUI:
    """Terminal user interface components for CLI mode."""
    
    @staticmethod
    def print_header(title):
        """Print a styled header."""
        console.print(f"[bold blue]{title}[/bold blue]", style="bold", justify="center")
        console.print("=" * 80, style="dim")
    
    @staticmethod
    def print_positions_table(positions):
        """
        Print a table of current positions.
        
        Args:
            positions: List of position dictionaries
        """
        table = Table(title="Current Positions")
        
        table.add_column("Symbol", style="cyan")
        table.add_column("Direction", style="magenta")
        table.add_column("Entry Price", style="blue")
        table.add_column("Current Price", style="blue")
        table.add_column("Size", style="yellow")
        table.add_column("P&L", style="green")
        table.add_column("P&L %", style="green")
        
        for pos in positions:
            pnl_style = "green" if pos["pnl"] >= 0 else "red"
            table.add_row(
                pos["symbol"],
                pos["direction"],
                f"{pos['entry_price']:.2f}",
                f"{pos['current_price']:.2f}",
                f"{pos['size']:.4f}",
                f"{pos['pnl']:.2f}", 
                f"{pos['pnl_percentage']:.2f}%",
                style=None if pos["pnl"] >= 0 else "red"
            )
        
        console.print(table)
    
    @staticmethod
    def print_trades_table(trades, limit=10):
        """
        Print a table of recent trades.
        
        Args:
            trades: List of trade dictionaries
            limit: Maximum number of trades to display
        """
        table = Table(title=f"Recent Trades (Last {min(limit, len(trades))})")
        
        table.add_column("Time", style="cyan")
        table.add_column("Symbol", style="cyan")
        table.add_column("Direction", style="magenta")
        table.add_column("Entry Price", style="blue")
        table.add_column("Exit Price", style="blue")
        table.add_column("Size", style="yellow")
        table.add_column("P&L", style="green")
        table.add_column("P&L %", style="green")
        
        for trade in trades[:limit]:
            pnl_style = "green" if trade["pnl"] >= 0 else "red"
            table.add_row(
                trade["exit_time"].strftime("%Y-%m-%d %H:%M"),
                trade["symbol"],
                trade["direction"],
                f"{trade['entry_price']:.2f}",
                f"{trade['exit_price']:.2f}",
                f"{trade['size']:.4f}",
                f"{trade['pnl']:.2f}",
                f"{trade['pnl_percentage']:.2f}%",
                style=None if trade["pnl"] >= 0 else "red"
            )
        
        console.print(table)
    
    @staticmethod
    def print_performance_metrics(metrics):
        """
        Print performance metrics in a panel.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        panel = Panel.fit(
            "\n".join([
                f"[bold]Total Return:[/bold] {metrics['total_return']:.2f}%",
                f"[bold]Win Rate:[/bold] {metrics['win_rate']:.2f}%",
                f"[bold]Profit Factor:[/bold] {metrics['profit_factor']:.2f}",
                f"[bold]Sharpe Ratio:[/bold] {metrics['sharpe_ratio']:.2f}",
                f"[bold]Max Drawdown:[/bold] {metrics['max_drawdown']:.2f}%"
            ]),
            title="Performance Metrics",
            border_style="blue"
        )
        console.print(panel)
    
    @staticmethod
    def create_progress_bar(total, description="Processing"):
        """
        Create a progress bar.
        
        Args:
            total: Total number of steps
            description: Description of the task
            
        Returns:
            Progress object and task ID
        """
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
        task_id = progress.add_task(description, total=total)
        return progress, task_id
    
    @staticmethod
    def create_dashboard_layout():
        """
        Create a rich layout for a live dashboard.
        
        Returns:
            Layout object configured for the dashboard
        """
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split(
            Layout(name="portfolio", ratio=2),
            Layout(name="positions", ratio=3)
        )
        
        layout["right"].split(
            Layout(name="performance", ratio=2),
            Layout(name="trades", ratio=3)
        )
        
        # Set up initial content
        layout["header"].update(
            Panel(
                Text("Abidance Trading Bot", style="bold blue", justify="center"),
                style="blue"
            )
        )
        
        layout["footer"].update(
            Panel(
                Text("Press Ctrl+C to exit", style="italic", justify="center"),
                style="dim"
            )
        )
        
        return layout
```

## 6.4 Configuration Interface

```python
# abidance/config/interface.py
import yaml
import json
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

class TimeFrame(str, Enum):
    """Supported timeframes for trading."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"

class RiskLevel(str, Enum):
    """Risk level settings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TradingMode(str, Enum):
    """Trading mode settings."""
    PAPER = "paper"
    LIVE = "live"

class StrategyConfig(BaseModel):
    """Configuration for a trading strategy."""
    name: str
    type: str
    symbol: str
    timeframe: TimeFrame
    parameters: Dict[str, Any]
    is_active: bool = False

class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_position_size: float = Field(..., gt=0, le=100)
    max_positions: int = Field(..., gt=0)
    max_drawdown: float = Field(..., gt=0, le=100)
    stop_loss_pct: Optional[float] = Field(None, gt=0, le=100)
    take_profit_pct: Optional[float] = Field(None, gt=0)
    risk_per_trade: float = Field(..., gt=0, le=100)
    risk_level: RiskLevel = RiskLevel.MEDIUM

class ExchangeConfig(BaseModel):
    """Exchange connection configuration."""
    name: str = "binance"
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    testnet: bool = True
    rate_limit_margin: float = Field(0.8, gt=0, le=1)

class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "sqlite"
    path: str = "data/trading.db"
    use_timescale: bool = False
    connection_string: Optional[str] = None

class WebConfig(BaseModel):
    """Web interface configuration."""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1024, le=65535)
    debug: bool = False
    open_browser: bool = True

class BotConfig(BaseModel):
    """Main bot configuration."""
    name: str
    trading_mode: TradingMode = TradingMode.PAPER
    exchange: ExchangeConfig
    risk: RiskConfig
    strategies: List[StrategyConfig]
    database: DatabaseConfig
    web: WebConfig
    log_level: str = "INFO"
    data_dir: str = "data"

class ConfigManager:
    """Manages loading and saving configuration."""
    
    @staticmethod
    def load_config(config_path: str) -> BotConfig:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Parsed BotConfig object
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_data = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")
        
        return BotConfig(**config_data)
    
    @staticmethod
    def save_config(config: BotConfig, config_path: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            config: BotConfig object
            config_path: Path to save the configuration file
        """
        config_data = config.dict()
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config_data, f, default_flow_style=False)
            elif config_path.endswith('.json'):
                json.dump(config_data, f, indent=2)
            else:
                raise ValueError("Configuration file must be YAML or JSON") 