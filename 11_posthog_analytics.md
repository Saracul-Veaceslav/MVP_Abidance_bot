# 11. PostHog Analytics

## 11.1 Configuration

### Setup
```typescript
// lib/posthog.ts
import posthog from "posthog-js"
import { PostHog } from "posthog-node"

// Client-side PostHog
export function initPostHogClient() {
  if (typeof window !== 'undefined') {
    posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY!, {
      api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://app.posthog.com',
      loaded: (posthog) => {
        if (process.env.NODE_ENV === 'development') {
          // Add debug logging in development
          posthog.debug()
        }
      }
    })
  }
}

// Server-side PostHog
export const serverPostHog = new PostHog(
  process.env.POSTHOG_KEY!,
  {
    host: process.env.POSTHOG_HOST || 'https://app.posthog.com'
  }
)
```

### Provider Component
```typescript
// components/providers/PostHogProvider.tsx
'use client'

import { useEffect } from "react"
import posthog from "posthog-js"
import { usePathname, useSearchParams } from "next/navigation"
import { initPostHogClient } from "@/lib/posthog"

export function PostHogProvider({
  children
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  
  useEffect(() => {
    initPostHogClient()
  }, [])
  
  useEffect(() => {
    if (pathname) {
      let url = window.origin + pathname
      if (searchParams?.toString()) {
        url = url + `?${searchParams.toString()}`
      }
      posthog.capture("$pageview", {
        $current_url: url
      })
    }
  }, [pathname, searchParams])
  
  return children
}
```

## 11.2 Event Tracking

### Event Types
```typescript
// types/analytics.ts
export type AnalyticsEvent =
  | {
      name: "strategy_created"
      properties: {
        strategyId: string
        strategyType: string
        symbol: string
        timeframe: string
      }
    }
  | {
      name: "trade_executed"
      properties: {
        tradeId: string
        symbol: string
        side: "buy" | "sell"
        amount: number
        price: number
        type: "market" | "limit"
        strategyId?: string
      }
    }
  | {
      name: "position_closed"
      properties: {
        positionId: string
        symbol: string
        profitLoss: number
        duration: number
        strategyId?: string
      }
    }
  | {
      name: "subscription_started"
      properties: {
        plan: string
        amount: number
        currency: string
        interval: "month" | "year"
      }
    }
```

### Analytics Hook
```typescript
// hooks/useAnalytics.ts
'use client'

import { useCallback } from "react"
import posthog from "posthog-js"
import type { AnalyticsEvent } from "@/types"

export function useAnalytics() {
  const trackEvent = useCallback(
    <T extends AnalyticsEvent>(
      event: T["name"],
      properties: T["properties"]
    ) => {
      posthog.capture(event, properties)
    },
    []
  )
  
  const identify = useCallback((userId: string, traits?: object) => {
    posthog.identify(userId, traits)
  }, [])
  
  return {
    trackEvent,
    identify
  }
}
```

### Server-Side Analytics
```typescript
// lib/analytics.ts
import { serverPostHog } from "@/lib/posthog"
import type { AnalyticsEvent } from "@/types"

export async function trackServerEvent<T extends AnalyticsEvent>(
  userId: string,
  event: T["name"],
  properties: T["properties"]
) {
  try {
    await serverPostHog.capture({
      distinctId: userId,
      event,
      properties
    })
  } catch (error) {
    console.error("Failed to track event:", error)
  }
}
```

## 11.3 Usage Examples

### Client-Side Tracking
```typescript
// components/strategies/CreateStrategyForm.tsx
'use client'

import { useAnalytics } from "@/hooks/useAnalytics"
import { createStrategyAction } from "@/actions/strategies"

export function CreateStrategyForm() {
  const { trackEvent } = useAnalytics()
  
  async function onSubmit(data: FormData) {
    const result = await createStrategyAction(data)
    
    if (result.isSuccess) {
      trackEvent("strategy_created", {
        strategyId: result.data.id,
        strategyType: data.type,
        symbol: data.symbol,
        timeframe: data.timeframe
      })
    }
  }
  
  return (
    // Form implementation
  )
}

// components/trading/TradeForm.tsx
'use client'

import { useAnalytics } from "@/hooks/useAnalytics"
import { executeTradeAction } from "@/actions/trading"

export function TradeForm() {
  const { trackEvent } = useAnalytics()
  
  async function onSubmit(data: FormData) {
    const result = await executeTradeAction(data)
    
    if (result.isSuccess) {
      trackEvent("trade_executed", {
        tradeId: result.data.id,
        symbol: data.symbol,
        side: data.side,
        amount: data.amount,
        price: data.price,
        type: data.type,
        strategyId: data.strategyId
      })
    }
  }
  
  return (
    // Form implementation
  )
}
```

### Server-Side Tracking
```typescript
// actions/positions.ts
'use server'

import { auth } from "@clerk/nextjs"
import { trackServerEvent } from "@/lib/analytics"
import { db } from "@/lib/db"
import { eq } from "drizzle-orm"
import { positions } from "@/db/schema"
import type { ActionState } from "@/types"

export async function closePositionAction(
  positionId: string
): Promise<ActionState<void>> {
  try {
    const { userId } = auth()
    
    if (!userId) {
      return {
        isSuccess: false,
        message: "Not authenticated"
      }
    }
    
    const position = await db.query.positions.findFirst({
      where: eq(positions.id, positionId)
    })
    
    if (!position) {
      return {
        isSuccess: false,
        message: "Position not found"
      }
    }
    
    // Close position logic here
    
    // Track the event
    await trackServerEvent(userId, "position_closed", {
      positionId,
      symbol: position.symbol,
      profitLoss: position.profitLoss,
      duration: Date.now() - position.openedAt.getTime(),
      strategyId: position.strategyId
    })
    
    return {
      isSuccess: true,
      message: "Position closed successfully"
    }
  } catch (error) {
    console.error("Error closing position:", error)
    return {
      isSuccess: false,
      message: "Failed to close position"
    }
  }
}
```

## 11.4 Feature Flags

### Feature Flag Types
```typescript
// types/feature-flags.ts
export interface FeatureFlags {
  advancedStrategies: boolean
  betaFeatures: boolean
  newUI: boolean
}

export type FeatureFlagKey = keyof FeatureFlags
```

### Feature Flag Hook
```typescript
// hooks/useFeatureFlags.ts
'use client'

import { useCallback } from "react"
import posthog from "posthog-js"
import type { FeatureFlagKey } from "@/types"

export function useFeatureFlags() {
  const getFeatureFlag = useCallback(
    (key: FeatureFlagKey, defaultValue: boolean = false): boolean => {
      return posthog.isFeatureEnabled(key) ?? defaultValue
    },
    []
  )
  
  const reloadFeatureFlags = useCallback(async () => {
    await posthog.reloadFeatureFlags()
  }, [])
  
  return {
    getFeatureFlag,
    reloadFeatureFlags
  }
}
```

### Feature Flag Usage
```typescript
// components/strategies/AdvancedStrategyCard.tsx
'use client'

import { useFeatureFlags } from "@/hooks/useFeatureFlags"

export function AdvancedStrategyCard() {
  const { getFeatureFlag } = useFeatureFlags()
  const showAdvancedFeatures = getFeatureFlag("advancedStrategies")
  
  if (!showAdvancedFeatures) {
    return null
  }
  
  return (
    <div className="rounded-lg border p-6">
      <h3 className="text-lg font-medium">
        Advanced Strategy Settings
      </h3>
      {/* Advanced features */}
    </div>
  )
}

// app/layout.tsx
import { headers } from "next/headers"
import { PostHogProvider } from "@/components/providers/PostHogProvider"

export default function RootLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <PostHogProvider>
          {children}
        </PostHogProvider>
      </body>
    </html>
  )
}
``` 