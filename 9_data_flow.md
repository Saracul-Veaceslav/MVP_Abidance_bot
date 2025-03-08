# 9. Data Flow

## 9.1 Server Actions

### Action Types
```typescript
// types/actions.ts
export type ActionState<T> = {
  isSuccess: boolean
  message: string
  data?: T
  error?: {
    code: string
    details?: unknown
  }
}

export type OptimisticAction<T> = {
  action: () => Promise<ActionState<T>>
  optimisticData: T
  revalidate?: string[]
}
```

### Action Implementation
```typescript
// actions/strategies.ts
'use server'

import { db } from "@/lib/db"
import { revalidatePath } from "next/cache"
import { strategiesTable } from "@/db/schema"
import { eq } from "drizzle-orm"
import type { ActionState } from "@/types"

export async function createStrategyAction(
  data: InsertStrategy
): Promise<ActionState<Strategy>> {
  try {
    const [strategy] = await db
      .insert(strategiesTable)
      .values(data)
      .returning()
    
    revalidatePath('/strategies')
    
    return {
      isSuccess: true,
      message: "Strategy created successfully",
      data: strategy
    }
  } catch (error) {
    return {
      isSuccess: false,
      message: "Failed to create strategy",
      error: {
        code: 'DB_ERROR',
        details: error
      }
    }
  }
}

export async function updateStrategyAction(
  id: string,
  data: Partial<InsertStrategy>
): Promise<ActionState<Strategy>> {
  try {
    const [strategy] = await db
      .update(strategiesTable)
      .set(data)
      .where(eq(strategiesTable.id, id))
      .returning()
    
    revalidatePath('/strategies')
    revalidatePath(`/strategies/${id}`)
    
    return {
      isSuccess: true,
      message: "Strategy updated successfully",
      data: strategy
    }
  } catch (error) {
    return {
      isSuccess: false,
      message: "Failed to update strategy",
      error: {
        code: 'DB_ERROR',
        details: error
      }
    }
  }
}
```

### Action Hooks
```typescript
// hooks/useAction.ts
'use client'

import { useState } from "react"
import { useRouter } from "next/navigation"
import type { ActionState, OptimisticAction } from "@/types"

export function useAction<T>() {
  const [isPending, setIsPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()
  
  async function execute(
    actionFn: () => Promise<ActionState<T>>
  ): Promise<ActionState<T>> {
    try {
      setIsPending(true)
      setError(null)
      
      const result = await actionFn()
      
      if (!result.isSuccess) {
        setError(result.message)
      }
      
      return result
    } catch (error) {
      setError("An unexpected error occurred")
      return {
        isSuccess: false,
        message: "An unexpected error occurred",
        error: {
          code: 'UNKNOWN_ERROR',
          details: error
        }
      }
    } finally {
      setIsPending(false)
    }
  }
  
  async function executeOptimistic<T>({
    action,
    optimisticData,
    revalidate
  }: OptimisticAction<T>): Promise<ActionState<T>> {
    // Start optimistic update
    setIsPending(true)
    setError(null)
    
    try {
      // Execute the action
      const result = await action()
      
      if (!result.isSuccess) {
        setError(result.message)
      } else if (revalidate) {
        // Revalidate paths on success
        revalidate.forEach(path => router.refresh())
      }
      
      return result
    } catch (error) {
      setError("An unexpected error occurred")
      return {
        isSuccess: false,
        message: "An unexpected error occurred",
        error: {
          code: 'UNKNOWN_ERROR',
          details: error
        }
      }
    } finally {
      setIsPending(false)
    }
  }
  
  return {
    execute,
    executeOptimistic,
    isPending,
    error
  }
}
```

## 9.2 Data Fetching

### Server-Side Data Fetching
```typescript
// app/strategies/page.tsx
import { Suspense } from "react"
import { getStrategies } from "@/actions/strategies"
import { StrategiesList } from "./_components/strategies-list"
import { StrategiesListSkeleton } from "./_components/strategies-list-skeleton"

export default async function StrategiesPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Trading Strategies</h1>
      
      <Suspense fallback={<StrategiesListSkeleton />}>
        <StrategiesLoader />
      </Suspense>
    </div>
  )
}

async function StrategiesLoader() {
  const { data: strategies } = await getStrategies()
  return <StrategiesList strategies={strategies} />
}
```

### Client-Side Data Fetching
```typescript
// hooks/useStrategies.ts
'use client'

import { useQuery } from "@tanstack/react-query"
import type { Strategy } from "@/types"

export function useStrategies() {
  return useQuery<Strategy[]>({
    queryKey: ['strategies'],
    queryFn: async () => {
      const response = await fetch('/api/strategies')
      if (!response.ok) {
        throw new Error('Failed to fetch strategies')
      }
      return response.json()
    }
  })
}

// components/strategies/StrategyCard.tsx
'use client'

import { useStrategies } from "@/hooks/useStrategies"
import { useAction } from "@/hooks/useAction"
import { updateStrategyAction } from "@/actions/strategies"

export function StrategyCard({ strategy }: { strategy: Strategy }) {
  const { execute } = useAction()
  const { data: strategies } = useStrategies()
  
  async function handleToggle() {
    await execute(() => 
      updateStrategyAction(strategy.id, {
        isActive: !strategy.isActive
      })
    )
  }
  
  return (
    <div className="p-4 border rounded-lg">
      <h3 className="font-medium">{strategy.name}</h3>
      <p className="text-sm text-gray-500">{strategy.description}</p>
      
      <div className="mt-4">
        <Switch
          checked={strategy.isActive}
          onCheckedChange={handleToggle}
        />
      </div>
    </div>
  )
}
```

## 9.3 Real-time Updates

### WebSocket Configuration
```typescript
// lib/websocket.ts
import { createClient } from "@supabase/supabase-js"

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export function subscribeToTrades(
  callback: (trade: Trade) => void
): () => void {
  const subscription = supabase
    .channel('trades')
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'trades'
      },
      (payload) => callback(payload.new as Trade)
    )
    .subscribe()
    
  return () => {
    subscription.unsubscribe()
  }
}

export function subscribeToPositions(
  callback: (position: Position) => void
): () => void {
  const subscription = supabase
    .channel('positions')
    .on(
      'postgres_changes',
      {
        event: '*',
        schema: 'public',
        table: 'positions'
      },
      (payload) => callback(payload.new as Position)
    )
    .subscribe()
    
  return () => {
    subscription.unsubscribe()
  }
}
```

### Real-time Hooks
```typescript
// hooks/useRealtimeData.ts
'use client'

import { useState, useEffect } from "react"
import { subscribeToTrades, subscribeToPositions } from "@/lib/websocket"
import type { Trade, Position } from "@/types"

export function useRealtimeTrades() {
  const [trades, setTrades] = useState<Trade[]>([])
  
  useEffect(() => {
    const unsubscribe = subscribeToTrades((trade) => {
      setTrades(prev => [trade, ...prev].slice(0, 100))
    })
    
    return () => {
      unsubscribe()
    }
  }, [])
  
  return trades
}

export function useRealtimePositions() {
  const [positions, setPositions] = useState<Position[]>([])
  
  useEffect(() => {
    const unsubscribe = subscribeToPositions((position) => {
      setPositions(prev => {
        const index = prev.findIndex(p => p.id === position.id)
        if (index === -1) {
          return [...prev, position]
        }
        const next = [...prev]
        next[index] = position
        return next
      })
    })
    
    return () => {
      unsubscribe()
    }
  }, [])
  
  return positions
}
```

### Real-time Components
```typescript
// components/dashboard/TradesFeed.tsx
'use client'

import { useRealtimeTrades } from "@/hooks/useRealtimeData"
import { formatCurrency } from "@/lib/format"

export function TradesFeed() {
  const trades = useRealtimeTrades()
  
  return (
    <div className="space-y-2">
      <h2 className="text-lg font-medium">Recent Trades</h2>
      
      <div className="divide-y">
        {trades.map(trade => (
          <div
            key={trade.id}
            className="py-2 flex items-center justify-between"
          >
            <div>
              <p className="font-medium">
                {trade.symbol}
              </p>
              <p className="text-sm text-gray-500">
                {trade.side} @ {formatCurrency(trade.price)}
              </p>
            </div>
            
            <div className="text-right">
              <p className="font-medium">
                {formatCurrency(trade.amount)}
              </p>
              <p className="text-sm text-gray-500">
                {new Date(trade.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// components/dashboard/PositionsTable.tsx
'use client'

import { useRealtimePositions } from "@/hooks/useRealtimeData"
import { formatCurrency, formatPercent } from "@/lib/format"

export function PositionsTable() {
  const positions = useRealtimePositions()
  
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-medium">Open Positions</h2>
      
      <div className="border rounded-lg divide-y">
        {positions.map(position => {
          const profitLoss = position.currentValue - position.entryValue
          const profitLossPercent = profitLoss / position.entryValue
          
          return (
            <div
              key={position.id}
              className="p-4 flex items-center justify-between"
            >
              <div>
                <p className="font-medium">
                  {position.symbol}
                </p>
                <p className="text-sm text-gray-500">
                  Entry @ {formatCurrency(position.entryPrice)}
                </p>
              </div>
              
              <div className="text-right">
                <p className={cn(
                  "font-medium",
                  profitLoss > 0 ? "text-green-600" : "text-red-600"
                )}>
                  {formatCurrency(profitLoss)}
                </p>
                <p className="text-sm text-gray-500">
                  {formatPercent(profitLossPercent)}
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
```

## 9.4 Data Validation

### Validation Schemas
```typescript
// lib/validations.ts
import { z } from "zod"

export const strategySchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  symbol: z.string().min(1),
  timeframe: z.enum(['1m', '5m', '15m', '1h', '4h', '1d']),
  parameters: z.record(z.unknown()),
  isActive: z.boolean().default(false)
})

export const tradeSchema = z.object({
  symbol: z.string().min(1),
  side: z.enum(['buy', 'sell']),
  type: z.enum(['market', 'limit']),
  price: z.number().positive(),
  amount: z.number().positive(),
  timestamp: z.string().datetime()
})

export const positionSchema = z.object({
  symbol: z.string().min(1),
  side: z.enum(['long', 'short']),
  entryPrice: z.number().positive(),
  currentPrice: z.number().positive(),
  size: z.number().positive(),
  leverage: z.number().min(1).max(100),
  stopLoss: z.number().positive().optional(),
  takeProfit: z.number().positive().optional()
})
```

### Validation Middleware
```typescript
// middleware/validate.ts
import { NextResponse } from "next/server"
import { z } from "zod"

export function validateRequest<T>(
  schema: z.Schema<T>,
  data: unknown
): NextResponse<T> | NextResponse<{ error: string }> {
  try {
    const validated = schema.parse(data)
    return NextResponse.json(validated)
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: error.errors[0].message },
        { status: 400 }
      )
    }
    return NextResponse.json(
      { error: "Invalid request data" },
      { status: 400 }
    )
  }
}

// app/api/strategies/route.ts
import { validateRequest } from "@/middleware/validate"
import { strategySchema } from "@/lib/validations"

export async function POST(request: Request) {
  const data = await request.json()
  
  const validationResult = validateRequest(
    strategySchema,
    data
  )
  
  if ('error' in validationResult) {
    return validationResult
  }
  
  // Process validated data
  const strategy = validationResult.data
  // ...
}
```

### Form Validation
```typescript
// components/strategies/StrategyForm.tsx
'use client'

import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { strategySchema } from "@/lib/validations"
import type { z } from "zod"

type FormData = z.infer<typeof strategySchema>

export function StrategyForm() {
  const form = useForm<FormData>({
    resolver: zodResolver(strategySchema),
    defaultValues: {
      isActive: false
    }
  })
  
  async function onSubmit(data: FormData) {
    // Handle form submission
  }
  
  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="space-y-4"
      >
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Strategy Name</FormLabel>
              <FormControl>
                <Input {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <FormField
          control={form.control}
          name="symbol"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Trading Symbol</FormLabel>
              <FormControl>
                <Input {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <FormField
          control={form.control}
          name="timeframe"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Timeframe</FormLabel>
              <Select
                onValueChange={field.onChange}
                defaultValue={field.value}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1m">1 Minute</SelectItem>
                  <SelectItem value="5m">5 Minutes</SelectItem>
                  <SelectItem value="15m">15 Minutes</SelectItem>
                  <SelectItem value="1h">1 Hour</SelectItem>
                  <SelectItem value="4h">4 Hours</SelectItem>
                  <SelectItem value="1d">1 Day</SelectItem>
                </SelectContent>
              </Select>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <Button type="submit">
          Create Strategy
        </Button>
      </form>
    </Form>
  )
}
``` 