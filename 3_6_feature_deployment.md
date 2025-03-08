# 3.6 Deployment and Operations

## User Story
As a system administrator, I want robust deployment and operational capabilities so that I can reliably run, monitor, and maintain the trading bot in production environments.

## Implementation Details

#### 3.6.1 Docker Containerization
- Container setup:
  - Multi-stage builds
  - Service isolation
  - Resource management
  
- Container orchestration:
  - Service dependencies
  - Health checks
  - Auto-recovery

#### Container Setup
```dockerfile
# Dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

# Create non-root user
RUN useradd -m -u 1000 trader
USER trader

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start command
CMD ["python", "-m", "abidance.main"]

# docker-compose.yml
version: '3.8'

services:
  trading-bot:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - database
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
          
  database:
    image: timescale/timescaledb:latest-pg14
    environment:
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
      
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  redis_data:
```

#### 3.6.2 Monitoring and Logging
- System monitoring:
  - Resource utilization
  - Service health
  - Performance metrics
  
- Application logging:
  - Structured logging
  - Log aggregation
  - Error tracking

#### System Monitoring
```python
# monitoring/metrics.py
from dataclasses import dataclass
from typing import Dict, List
import psutil
import logging
from prometheus_client import Counter, Gauge, Histogram

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int

class MetricsCollector:
    def __init__(self):
        # Prometheus metrics
        self.trade_counter = Counter(
            'trades_total',
            'Total number of trades executed',
            ['strategy', 'symbol', 'direction']
        )
        
        self.position_gauge = Gauge(
            'active_positions',
            'Number of active positions',
            ['strategy']
        )
        
        self.execution_latency = Histogram(
            'trade_execution_latency_seconds',
            'Time taken to execute trades',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        return SystemMetrics(
            cpu_usage=psutil.cpu_percent(interval=1),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io={
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            },
            process_count=len(psutil.pids())
        )
        
    def record_trade(
        self,
        strategy: str,
        symbol: str,
        direction: str
    ) -> None:
        """Record a completed trade."""
        self.trade_counter.labels(
            strategy=strategy,
            symbol=symbol,
            direction=direction
        ).inc()
        
    def update_positions(
        self,
        strategy: str,
        count: int
    ) -> None:
        """Update number of active positions."""
        self.position_gauge.labels(strategy=strategy).set(count)
        
    def measure_execution_time(self):
        """Context manager for measuring execution time."""
        return self.execution_latency.time()

# monitoring/logging.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Add JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
        
    def log(
        self,
        level: int,
        message: str,
        **kwargs: Any
    ) -> None:
        """Log a message with structured data."""
        extra = {
            'timestamp': datetime.utcnow().isoformat(),
            'data': kwargs
        }
        self.logger.log(level, message, extra=extra)
        
    def info(self, message: str, **kwargs: Any) -> None:
        self.log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs: Any) -> None:
        self.log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs: Any) -> None:
        self.log(logging.ERROR, message, **kwargs)
        
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        data = {
            'timestamp': getattr(
                record,
                'timestamp',
                datetime.utcnow().isoformat()
            ),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name
        }
        
        if hasattr(record, 'data'):
            data.update(record.data)
            
        return json.dumps(data)
```

#### 3.6.3 Configuration Management
- Environment configuration:
  - Environment variables
  - Configuration files
  - Secret management
  
- Dynamic configuration:
  - Hot reloading
  - Feature flags
  - Strategy parameters

#### Environment Configuration
```python
# config/settings.py
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import os
from pydantic import BaseSettings, SecretStr

class ExchangeSettings(BaseSettings):
    api_key: SecretStr
    api_secret: SecretStr
    test_mode: bool = True
    rate_limit: int = 1200
    
    class Config:
        env_prefix = 'BINANCE_'

class DatabaseSettings(BaseSettings):
    url: str
    pool_size: int = 5
    max_overflow: int = 10
    
    class Config:
        env_prefix = 'DB_'

class RiskSettings(BaseSettings):
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.03
    max_position_size: float = 0.25
    risk_per_trade: float = 0.01
    
    class Config:
        env_prefix = 'RISK_'

class Settings(BaseSettings):
    exchange: ExchangeSettings = ExchangeSettings()
    database: DatabaseSettings = DatabaseSettings()
    risk: RiskSettings = RiskSettings()
    
    environment: str = 'development'
    log_level: str = 'INFO'
    enable_metrics: bool = True
    
    @classmethod
    def load(cls) -> 'Settings':
        """Load settings from environment and config files."""
        # Load from environment
        settings = cls()
        
        # Load from config file
        config_path = os.getenv(
            'CONFIG_PATH',
            Path(__file__).parent / 'config.yml'
        )
        
        if Path(config_path).exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
                for key, value in config_data.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)
                        
        return settings

# config/feature_flags.py
from typing import Dict, Any
import redis
from functools import lru_cache

class FeatureFlags:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.cache_ttl = 60  # seconds
        
    @lru_cache(maxsize=100)
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        value = self.redis.get(f"feature:{feature}")
        return value == b"1"
        
    def set_feature(self, feature: str, enabled: bool) -> None:
        """Enable or disable a feature."""
        self.redis.set(f"feature:{feature}", "1" if enabled else "0")
        self.is_enabled.cache_clear()
        
    def get_all_features(self) -> Dict[str, bool]:
        """Get all feature flags."""
        features = {}
        for key in self.redis.scan_iter("feature:*"):
            feature = key.decode('utf-8').split(':')[1]
            features[feature] = self.is_enabled(feature)
        return features
```

#### 3.6.4 Backup and Recovery
- Data backup:
  - Database backups
  - Configuration backups
  - State persistence
  
- Recovery procedures:
  - System recovery
  - Data restoration
  - State reconstruction

#### Data Backup
```python
# operations/backup.py
from typing import List, Optional
import subprocess
import boto3
from datetime import datetime
import logging
from pathlib import Path

class BackupManager:
    def __init__(
        self,
        database_url: str,
        backup_path: Path,
        s3_bucket: Optional[str] = None
    ):
        self.database_url = database_url
        self.backup_path = backup_path
        self.s3_bucket = s3_bucket
        self.logger = logging.getLogger(__name__)
        
    def create_database_backup(self) -> Path:
        """Create a database backup."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_path / f"db_backup_{timestamp}.sql"
        
        try:
            # Create backup using pg_dump
            subprocess.run([
                'pg_dump',
                self.database_url,
                '-F', 'c',  # Custom format
                '-f', str(backup_file)
            ], check=True)
            
            self.logger.info(
                f"Created database backup: {backup_file}"
            )
            return backup_file
            
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Database backup failed: {str(e)}"
            )
            raise
            
    def upload_backup(self, backup_file: Path) -> None:
        """Upload backup to S3."""
        if not self.s3_bucket:
            return
            
        try:
            s3 = boto3.client('s3')
            s3.upload_file(
                str(backup_file),
                self.s3_bucket,
                f"backups/{backup_file.name}"
            )
            
            self.logger.info(
                f"Uploaded backup to S3: {backup_file.name}"
            )
            
        except Exception as e:
            self.logger.error(
                f"Backup upload failed: {str(e)}"
            )
            raise
            
    def cleanup_old_backups(
        self,
        keep_last: int = 7
    ) -> None:
        """Clean up old backup files."""
        backup_files = sorted(
            self.backup_path.glob('db_backup_*.sql'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Keep only the specified number of recent backups
        for backup_file in backup_files[keep_last:]:
            try:
                backup_file.unlink()
                self.logger.info(
                    f"Deleted old backup: {backup_file}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to delete backup {backup_file}: {str(e)}"
                )

# operations/recovery.py
class RecoveryManager:
    def __init__(
        self,
        database_url: str,
        backup_path: Path,
        s3_bucket: Optional[str] = None
    ):
        self.database_url = database_url
        self.backup_path = backup_path
        self.s3_bucket = s3_bucket
        self.logger = logging.getLogger(__name__)
        
    def restore_database(
        self,
        backup_file: Path
    ) -> None:
        """Restore database from backup."""
        try:
            # Restore using pg_restore
            subprocess.run([
                'pg_restore',
                '-d', self.database_url,
                '-c',  # Clean (drop) database objects before recreating
                str(backup_file)
            ], check=True)
            
            self.logger.info(
                f"Restored database from backup: {backup_file}"
            )
            
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Database restore failed: {str(e)}"
            )
            raise
            
    def download_backup(
        self,
        backup_name: str
    ) -> Path:
        """Download backup from S3."""
        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
            
        backup_file = self.backup_path / backup_name
        
        try:
            s3 = boto3.client('s3')
            s3.download_file(
                self.s3_bucket,
                f"backups/{backup_name}",
                str(backup_file)
            )
            
            self.logger.info(
                f"Downloaded backup from S3: {backup_name}"
            )
            return backup_file
            
        except Exception as e:
            self.logger.error(
                f"Backup download failed: {str(e)}"
            )
            raise
            
    def verify_database(self) -> bool:
        """Verify database integrity."""
        try:
            # Run basic verification queries
            with get_db_connection() as conn:
                # Check tables exist
                tables = conn.execute(
                    "SELECT tablename FROM pg_tables"
                ).fetchall()
                
                # Check recent data exists
                recent_trades = conn.execute(
                    "SELECT COUNT(*) FROM trades "
                    "WHERE created_at > NOW() - INTERVAL '1 day'"
                ).scalar()
                
                return len(tables) > 0 and recent_trades > 0
                
        except Exception as e:
            self.logger.error(
                f"Database verification failed: {str(e)}"
            )
            return False
```

### Error Handling and Edge Cases
- Deployment failures:
  - Rollback procedures
  - State recovery
  - Data consistency
  
- System outages:
  - Failover procedures
  - Service recovery
  - Data reconciliation
  
- Resource constraints:
  - Resource scaling
  - Load balancing
  - Performance optimization

### Testing Strategy

### Unit Tests
```python
# tests/unit/operations/test_backup.py
class TestBackupManager:
    """
    Feature: Database Backup Management
    """
    
    def setup_method(self):
        self.backup_path = Path('/tmp/test_backups')
        self.backup_path.mkdir(exist_ok=True)
        
        self.manager = BackupManager(
            database_url="postgresql://test:test@localhost:5432/test",
            backup_path=self.backup_path
        )
        
    def teardown_method(self):
        # Clean up test backups
        for file in self.backup_path.glob('*'):
            file.unlink()
        self.backup_path.rmdir()
        
    def test_create_backup(self):
        """
        Scenario: Create database backup
          Given a configured backup manager
          When creating a new backup
          Then a backup file should be created
          And the backup file should be valid
        """
        backup_file = self.manager.create_database_backup()
        
        assert backup_file.exists()
        assert backup_file.stat().st_size > 0
        
    def test_cleanup_old_backups(self):
        """
        Scenario: Clean up old backups
          Given multiple existing backup files
          When cleaning up old backups
          Then only the specified number of recent backups should be kept
        """
        # Create test backup files
        for i in range(10):
            backup_file = self.backup_path / f"db_backup_test_{i}.sql"
            backup_file.touch()
            
        self.manager.cleanup_old_backups(keep_last=5)
        
        remaining_files = list(self.backup_path.glob('*.sql'))
        assert len(remaining_files) == 5

# tests/unit/operations/test_recovery.py
class TestRecoveryManager:
    """
    Feature: Database Recovery Management
    """
    
    def setup_method(self):
        self.backup_path = Path('/tmp/test_recovery')
        self.backup_path.mkdir(exist_ok=True)
        
        self.manager = RecoveryManager(
            database_url="postgresql://test:test@localhost:5432/test",
            backup_path=self.backup_path
        )
        
    def test_verify_database(self):
        """
        Scenario: Verify database integrity
          Given a restored database
          When verifying the database
          Then the verification should pass if the database is valid
        """
        # Create test data
        with get_db_connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS trades ("
                "id SERIAL PRIMARY KEY, "
                "created_at TIMESTAMP DEFAULT NOW()"
                ")"
            )
            conn.execute(
                "INSERT INTO trades DEFAULT VALUES"
            )
            
        assert self.manager.verify_database() is True
        
    def test_restore_database(self):
        """
        Scenario: Restore database from backup
          Given a valid backup file
          When restoring the database
          Then the database should be restored successfully
          And the data should be accessible
        """
        # Create test backup
        backup_file = self.backup_path / "test_backup.sql"
        
        # Perform restore
        self.manager.restore_database(backup_file)
        
        # Verify restoration
        assert self.manager.verify_database() is True
``` 