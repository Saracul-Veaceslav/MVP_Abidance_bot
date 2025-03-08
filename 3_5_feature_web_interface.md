# 3.5 Web Interface

## User Story
As a trader, I want a modern web interface to monitor my trading bot's performance, manage strategies, and control trading operations in real-time.

## Implementation Details

### 3.5.1 RESTful API

#### Core Endpoints
```python
# api/routes.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import jwt

app = FastAPI(title="Abidance Trading Bot API")
security = HTTPBearer()

# Authentication middleware
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict:
    """Validate JWT token and return user info."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

# Strategy routes
@app.get("/api/v1/strategies")
async def list_strategies(
    current_user: Dict = Depends(get_current_user)
) -> List[Dict]:
    """List all available trading strategies."""
    try:
        strategies = await strategy_service.list_strategies(
            user_id=current_user['id']
        )
        return strategies
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/api/v1/strategies/{strategy_id}/activate")
async def activate_strategy(
    strategy_id: str,
    parameters: Dict,
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Activate a trading strategy."""
    try:
        result = await strategy_service.activate_strategy(
            strategy_id=strategy_id,
            parameters=parameters,
            user_id=current_user['id']
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Position routes
@app.get("/api/v1/positions")
async def list_positions(
    current_user: Dict = Depends(get_current_user)
) -> List[Dict]:
    """List all open positions."""
    try:
        positions = await position_service.list_positions(
            user_id=current_user['id']
        )
        return positions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/api/v1/positions/{position_id}/close")
async def close_position(
    position_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Close a specific position."""
    try:
        result = await position_service.close_position(
            position_id=position_id,
            user_id=current_user['id']
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Performance routes
@app.get("/api/v1/performance/summary")
async def get_performance_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Get performance summary for the specified period."""
    try:
        summary = await performance_service.get_summary(
            user_id=current_user['id'],
            start_date=start_date,
            end_date=end_date
        )
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/v1/performance/trades")
async def list_trades(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    strategy_id: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
) -> List[Dict]:
    """List trades for the specified period and strategy."""
    try:
        trades = await performance_service.list_trades(
            user_id=current_user['id'],
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        return trades
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
```

### 3.5.2 Web Dashboard

#### Portfolio Summary
```typescript
// components/Dashboard/PortfolioSummary.tsx
import React from 'react'
import { Card, Grid, Typography } from '@mui/material'
import { formatCurrency, formatPercentage } from '@/utils/format'

interface PortfolioMetrics {
  totalValue: number
  dailyPnL: number
  dailyReturn: number
  totalReturn: number
  winRate: number
  sharpeRatio: number
}

interface PortfolioSummaryProps {
  metrics: PortfolioMetrics
  isLoading: boolean
}

export const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({
  metrics,
  isLoading
}) => {
  if (isLoading) {
    return <LoadingSkeleton />
  }
  
  return (
    <Card>
      <Grid container spacing={2} p={2}>
        <Grid item xs={12} md={4}>
          <Typography variant="subtitle2" color="textSecondary">
            Portfolio Value
          </Typography>
          <Typography variant="h4">
            {formatCurrency(metrics.totalValue)}
          </Typography>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Typography variant="subtitle2" color="textSecondary">
            Daily P&L
          </Typography>
          <Typography 
            variant="h4"
            color={metrics.dailyPnL >= 0 ? 'success.main' : 'error.main'}
          >
            {formatCurrency(metrics.dailyPnL)}
            <Typography
              component="span"
              variant="body2"
              color="textSecondary"
              ml={1}
            >
              ({formatPercentage(metrics.dailyReturn)})
            </Typography>
          </Typography>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Typography variant="subtitle2" color="textSecondary">
            Total Return
          </Typography>
          <Typography
            variant="h4"
            color={metrics.totalReturn >= 0 ? 'success.main' : 'error.main'}
          >
            {formatPercentage(metrics.totalReturn)}
          </Typography>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Typography variant="subtitle2" color="textSecondary">
            Win Rate
          </Typography>
          <Typography variant="h4">
            {formatPercentage(metrics.winRate)}
          </Typography>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Typography variant="subtitle2" color="textSecondary">
            Sharpe Ratio
          </Typography>
          <Typography variant="h4">
            {metrics.sharpeRatio.toFixed(2)}
          </Typography>
        </Grid>
      </Grid>
    </Card>
  )
}
```

#### Strategy List
```typescript
// components/Dashboard/StrategyList.tsx
import React from 'react'
import {
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  IconButton,
  Chip
} from '@mui/material'
import { PlayArrow, Stop, Settings } from '@mui/icons-material'

interface Strategy {
  id: string
  name: string
  isActive: boolean
  dailyPnL: number
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
}

interface StrategyListProps {
  strategies: Strategy[]
  onActivate: (id: string) => void
  onDeactivate: (id: string) => void
  onConfigure: (id: string) => void
}

export const StrategyList: React.FC<StrategyListProps> = ({
  strategies,
  onActivate,
  onDeactivate,
  onConfigure
}) => {
  return (
    <Card>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Strategy</TableCell>
            <TableCell>Status</TableCell>
            <TableCell align="right">Daily P&L</TableCell>
            <TableCell align="right">Total Return</TableCell>
            <TableCell align="right">Sharpe Ratio</TableCell>
            <TableCell align="right">Max Drawdown</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {strategies.map((strategy) => (
            <TableRow key={strategy.id}>
              <TableCell>{strategy.name}</TableCell>
              <TableCell>
                <Chip
                  label={strategy.isActive ? 'Active' : 'Inactive'}
                  color={strategy.isActive ? 'success' : 'default'}
                  size="small"
                />
              </TableCell>
              <TableCell align="right">
                <Typography
                  color={strategy.dailyPnL >= 0 ? 'success.main' : 'error.main'}
                >
                  {formatCurrency(strategy.dailyPnL)}
                </Typography>
              </TableCell>
              <TableCell align="right">
                {formatPercentage(strategy.totalReturn)}
              </TableCell>
              <TableCell align="right">
                {strategy.sharpeRatio.toFixed(2)}
              </TableCell>
              <TableCell align="right">
                {formatPercentage(strategy.maxDrawdown)}
              </TableCell>
              <TableCell align="right">
                <IconButton
                  onClick={() => 
                    strategy.isActive
                      ? onDeactivate(strategy.id)
                      : onActivate(strategy.id)
                  }
                  color={strategy.isActive ? 'error' : 'success'}
                >
                  {strategy.isActive ? <Stop /> : <PlayArrow />}
                </IconButton>
                <IconButton
                  onClick={() => onConfigure(strategy.id)}
                  color="primary"
                >
                  <Settings />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  )
}
```

### 3.5.3 Real-time Updates

#### WebSocket Integration
```typescript
// services/websocket.ts
import { io, Socket } from 'socket.io-client'
import { create } from 'zustand'

interface WebSocketStore {
  socket: Socket | null
  isConnected: boolean
  connect: (token: string) => void
  disconnect: () => void
}

export const useWebSocket = create<WebSocketStore>((set) => ({
  socket: null,
  isConnected: false,
  
  connect: (token: string) => {
    const socket = io(process.env.NEXT_PUBLIC_WS_URL!, {
      auth: { token },
      transports: ['websocket']
    })
    
    socket.on('connect', () => {
      set({ isConnected: true })
    })
    
    socket.on('disconnect', () => {
      set({ isConnected: false })
    })
    
    set({ socket })
  },
  
  disconnect: () => {
    set((state) => {
      state.socket?.disconnect()
      return { socket: null, isConnected: false }
    })
  }
}))

// components/LiveUpdates/PriceStream.tsx
import React, { useEffect, useState } from 'react'
import { useWebSocket } from '@/services/websocket'

interface PriceUpdate {
  symbol: string
  price: number
  timestamp: number
}

interface PriceStreamProps {
  symbols: string[]
  onUpdate: (update: PriceUpdate) => void
}

export const PriceStream: React.FC<PriceStreamProps> = ({
  symbols,
  onUpdate
}) => {
  const { socket, isConnected } = useWebSocket()
  
  useEffect(() => {
    if (!socket || !isConnected) return
    
    // Subscribe to price updates
    socket.emit('subscribe', { channel: 'prices', symbols })
    
    // Handle price updates
    socket.on('price', (update: PriceUpdate) => {
      onUpdate(update)
    })
    
    return () => {
      socket.emit('unsubscribe', { channel: 'prices', symbols })
      socket.off('price')
    }
  }, [socket, isConnected, symbols])
  
  return null
}

// components/LiveUpdates/NotificationCenter.tsx
import React, { useEffect } from 'react'
import { useWebSocket } from '@/services/websocket'
import { useSnackbar } from 'notistack'

interface Notification {
  type: 'success' | 'error' | 'warning' | 'info'
  message: string
  timestamp: number
}

export const NotificationCenter: React.FC = () => {
  const { socket, isConnected } = useWebSocket()
  const { enqueueSnackbar } = useSnackbar()
  
  useEffect(() => {
    if (!socket || !isConnected) return
    
    // Subscribe to notifications
    socket.emit('subscribe', { channel: 'notifications' })
    
    // Handle notifications
    socket.on('notification', (notification: Notification) => {
      enqueueSnackbar(notification.message, {
        variant: notification.type,
        autoHideDuration: 5000
      })
    })
    
    return () => {
      socket.emit('unsubscribe', { channel: 'notifications' })
      socket.off('notification')
    }
  }, [socket, isConnected])
  
  return null
}
```

## Error Handling and Edge Cases

### API Error Handling
- Input validation
- Error responses
- Rate limit handling

### WebSocket Reliability
- Connection recovery
- Message ordering
- State synchronization

### UI Error States
- Loading states
- Error boundaries
- Fallback content

## Testing Strategy

### Unit Tests
```typescript
// tests/unit/api/test_routes.py
class TestStrategyRoutes:
    """
    Feature: Strategy Management API
    """
    
    async def test_list_strategies(self, client, auth_headers):
        """
        Scenario: List available strategies
          Given an authenticated user
          When requesting the list of strategies
          Then the response should include all user's strategies
        """
        response = await client.get(
            "/api/v1/strategies",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        
    async def test_activate_strategy(self, client, auth_headers):
        """
        Scenario: Activate a trading strategy
          Given an authenticated user
          And a valid strategy configuration
          When activating the strategy
          Then the strategy should be started successfully
        """
        strategy_id = "test-strategy"
        parameters = {
            "risk_per_trade": 0.01,
            "max_positions": 5
        }
        
        response = await client.post(
            f"/api/v1/strategies/{strategy_id}/activate",
            json=parameters,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "active"

# tests/unit/components/test_portfolio_summary.tsx
import { render, screen } from '@testing-library/react'
import { PortfolioSummary } from '@/components/Dashboard/PortfolioSummary'

describe('PortfolioSummary', () => {
  const mockMetrics = {
    totalValue: 100000,
    dailyPnL: 1500,
    dailyReturn: 0.015,
    totalReturn: 0.25,
    winRate: 0.65,
    sharpeRatio: 1.8
  }
  
  it('displays portfolio metrics correctly', () => {
    render(<PortfolioSummary metrics={mockMetrics} isLoading={false} />)
    
    expect(screen.getByText('$100,000.00')).toBeInTheDocument()
    expect(screen.getByText('$1,500.00')).toBeInTheDocument()
    expect(screen.getByText('(1.50%)')).toBeInTheDocument()
    expect(screen.getByText('25.00%')).toBeInTheDocument()
    expect(screen.getByText('65.00%')).toBeInTheDocument()
    expect(screen.getByText('1.80')).toBeInTheDocument()
  })
  
  it('shows loading skeleton when loading', () => {
    render(<PortfolioSummary metrics={mockMetrics} isLoading={true} />)
    expect(screen.getByTestId('loading-skeleton')).toBeInTheDocument()
  })
})

# tests/unit/services/test_websocket.ts
import { renderHook, act } from '@testing-library/react-hooks'
import { useWebSocket } from '@/services/websocket'
import { io } from 'socket.io-client'

jest.mock('socket.io-client')

describe('WebSocket Store', () => {
  beforeEach(() => {
    (io as jest.Mock).mockClear()
  })
  
  it('connects to websocket server', () => {
    const { result } = renderHook(() => useWebSocket())
    
    act(() => {
      result.current.connect('test-token')
    })
    
    expect(io).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        auth: { token: 'test-token' },
        transports: ['websocket']
      })
    )
  })
  
  it('updates connection status on connect/disconnect', () => {
    const mockSocket = {
      on: jest.fn(),
      disconnect: jest.fn()
    }
    
    (io as jest.Mock).mockReturnValue(mockSocket)
    
    const { result } = renderHook(() => useWebSocket())
    
    act(() => {
      result.current.connect('test-token')
    })
    
    // Simulate connect event
    const connectHandler = mockSocket.on.mock.calls.find(
      call => call[0] === 'connect'
    )[1]
    act(() => {
      connectHandler()
    })
    
    expect(result.current.isConnected).toBe(true)
    
    // Simulate disconnect event
    const disconnectHandler = mockSocket.on.mock.calls.find(
      call => call[0] === 'disconnect'
    )[1]
    act(() => {
      disconnectHandler()
    })
    
    expect(result.current.isConnected).toBe(false)
  })
})
``` 