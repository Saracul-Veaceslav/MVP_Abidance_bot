# 10. Stripe Integration

## 10.1 Configuration

### Environment Setup
```typescript
// lib/stripe.ts
import Stripe from "stripe"

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: "2023-10-16",
  typescript: true
})

export const STRIPE_WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET!
```

### Price Configuration
```typescript
// config/stripe.ts
export const STRIPE_PLANS = {
  PRO: {
    name: "Pro Plan",
    description: "Advanced trading features and priority support",
    priceId: process.env.STRIPE_PRO_PRICE_ID!,
    features: [
      "Unlimited trading strategies",
      "Real-time market data",
      "Advanced analytics",
      "Priority support"
    ]
  },
  ENTERPRISE: {
    name: "Enterprise Plan",
    description: "Custom solutions for large-scale trading operations",
    priceId: process.env.STRIPE_ENTERPRISE_PRICE_ID!,
    features: [
      "Custom strategy development",
      "Dedicated support team",
      "API access",
      "Advanced risk management"
    ]
  }
} as const

export type PlanType = keyof typeof STRIPE_PLANS
```

## 10.2 Checkout Flow

### Checkout Actions
```typescript
// actions/stripe.ts
'use server'

import { stripe } from "@/lib/stripe"
import { auth } from "@clerk/nextjs"
import { db } from "@/lib/db"
import { eq } from "drizzle-orm"
import { users } from "@/db/schema"
import { STRIPE_PLANS, type PlanType } from "@/config/stripe"
import type { ActionState } from "@/types"

export async function createCheckoutSession(
  plan: PlanType
): Promise<ActionState<{ url: string }>> {
  try {
    const { userId } = auth()
    
    if (!userId) {
      return {
        isSuccess: false,
        message: "Not authenticated"
      }
    }
    
    // Get or create Stripe customer
    const user = await db.query.users.findFirst({
      where: eq(users.id, userId)
    })
    
    let customerId = user?.stripeCustomerId
    
    if (!customerId) {
      const customer = await stripe.customers.create({
        metadata: {
          userId
        }
      })
      
      await db
        .update(users)
        .set({ stripeCustomerId: customer.id })
        .where(eq(users.id, userId))
      
      customerId = customer.id
    }
    
    // Create checkout session
    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      line_items: [
        {
          price: STRIPE_PLANS[plan].priceId,
          quantity: 1
        }
      ],
      mode: "subscription",
      success_url: `${process.env.NEXT_PUBLIC_APP_URL}/settings/billing?success=true`,
      cancel_url: `${process.env.NEXT_PUBLIC_APP_URL}/settings/billing?canceled=true`,
      metadata: {
        userId,
        plan
      }
    })
    
    if (!session.url) {
      throw new Error("Failed to create checkout session")
    }
    
    return {
      isSuccess: true,
      message: "Checkout session created",
      data: {
        url: session.url
      }
    }
  } catch (error) {
    console.error("Error creating checkout session:", error)
    return {
      isSuccess: false,
      message: "Failed to create checkout session"
    }
  }
}

export async function createBillingPortalSession(
): Promise<ActionState<{ url: string }>> {
  try {
    const { userId } = auth()
    
    if (!userId) {
      return {
        isSuccess: false,
        message: "Not authenticated"
      }
    }
    
    const user = await db.query.users.findFirst({
      where: eq(users.id, userId)
    })
    
    if (!user?.stripeCustomerId) {
      return {
        isSuccess: false,
        message: "No billing information found"
      }
    }
    
    const session = await stripe.billingPortal.sessions.create({
      customer: user.stripeCustomerId,
      return_url: `${process.env.NEXT_PUBLIC_APP_URL}/settings/billing`
    })
    
    return {
      isSuccess: true,
      message: "Billing portal session created",
      data: {
        url: session.url
      }
    }
  } catch (error) {
    console.error("Error creating billing portal session:", error)
    return {
      isSuccess: false,
      message: "Failed to create billing portal session"
    }
  }
}
```

### Checkout Components
```typescript
// components/billing/PlanCard.tsx
'use client'

import { useAction } from "@/hooks/useAction"
import { createCheckoutSession } from "@/actions/stripe"
import { STRIPE_PLANS, type PlanType } from "@/config/stripe"

interface PlanCardProps {
  plan: PlanType
}

export function PlanCard({ plan }: PlanCardProps) {
  const { execute, isPending } = useAction()
  
  async function handleUpgrade() {
    const result = await execute(() => createCheckoutSession(plan))
    
    if (result.isSuccess && result.data) {
      window.location.href = result.data.url
    }
  }
  
  const planDetails = STRIPE_PLANS[plan]
  
  return (
    <div className="rounded-lg border p-6">
      <h3 className="text-lg font-medium">{planDetails.name}</h3>
      <p className="mt-2 text-gray-500">{planDetails.description}</p>
      
      <ul className="mt-4 space-y-2">
        {planDetails.features.map(feature => (
          <li key={feature} className="flex items-center">
            <CheckIcon className="mr-2 h-4 w-4 text-green-500" />
            <span>{feature}</span>
          </li>
        ))}
      </ul>
      
      <Button
        className="mt-6 w-full"
        onClick={handleUpgrade}
        disabled={isPending}
      >
        {isPending ? "Loading..." : "Upgrade"}
      </Button>
    </div>
  )
}

// components/billing/ManageSubscriptionButton.tsx
'use client'

import { useAction } from "@/hooks/useAction"
import { createBillingPortalSession } from "@/actions/stripe"

export function ManageSubscriptionButton() {
  const { execute, isPending } = useAction()
  
  async function handleManage() {
    const result = await execute(createBillingPortalSession)
    
    if (result.isSuccess && result.data) {
      window.location.href = result.data.url
    }
  }
  
  return (
    <Button
      variant="outline"
      onClick={handleManage}
      disabled={isPending}
    >
      {isPending ? "Loading..." : "Manage Subscription"}
    </Button>
  )
}
```

## 10.3 Webhook Handling

### Webhook Route
```typescript
// app/api/webhook/stripe/route.ts
import { headers } from "next/headers"
import { NextResponse } from "next/server"
import { stripe, STRIPE_WEBHOOK_SECRET } from "@/lib/stripe"
import { db } from "@/lib/db"
import { eq } from "drizzle-orm"
import { users, subscriptions } from "@/db/schema"
import type Stripe from "stripe"

async function handleSubscriptionCreated(
  subscription: Stripe.Subscription
) {
  const userId = subscription.metadata.userId
  
  if (!userId) {
    throw new Error("No user ID in subscription metadata")
  }
  
  await db.insert(subscriptions).values({
    userId,
    stripeSubscriptionId: subscription.id,
    stripePriceId: subscription.items.data[0].price.id,
    stripeCurrentPeriodEnd: new Date(
      subscription.current_period_end * 1000
    )
  })
}

async function handleSubscriptionUpdated(
  subscription: Stripe.Subscription
) {
  const userId = subscription.metadata.userId
  
  if (!userId) {
    throw new Error("No user ID in subscription metadata")
  }
  
  await db
    .update(subscriptions)
    .set({
      stripePriceId: subscription.items.data[0].price.id,
      stripeCurrentPeriodEnd: new Date(
        subscription.current_period_end * 1000
      )
    })
    .where(eq(subscriptions.stripeSubscriptionId, subscription.id))
}

async function handleSubscriptionDeleted(
  subscription: Stripe.Subscription
) {
  await db
    .delete(subscriptions)
    .where(eq(subscriptions.stripeSubscriptionId, subscription.id))
}

export async function POST(request: Request) {
  const body = await request.text()
  const signature = headers().get("Stripe-Signature")
  
  if (!signature) {
    return new Response("No signature", { status: 400 })
  }
  
  let event: Stripe.Event
  
  try {
    event = stripe.webhooks.constructEvent(
      body,
      signature,
      STRIPE_WEBHOOK_SECRET
    )
  } catch (error) {
    console.error("Error verifying webhook signature:", error)
    return new Response("Invalid signature", { status: 400 })
  }
  
  try {
    switch (event.type) {
      case "customer.subscription.created":
        await handleSubscriptionCreated(
          event.data.object as Stripe.Subscription
        )
        break
      
      case "customer.subscription.updated":
        await handleSubscriptionUpdated(
          event.data.object as Stripe.Subscription
        )
        break
      
      case "customer.subscription.deleted":
        await handleSubscriptionDeleted(
          event.data.object as Stripe.Subscription
        )
        break
    }
    
    return NextResponse.json({ received: true })
  } catch (error) {
    console.error("Error handling webhook event:", error)
    return new Response("Webhook handler failed", { status: 500 })
  }
}
```

## 10.4 Subscription Management

### Subscription Schema
```typescript
// db/schema/subscriptions-schema.ts
import {
  pgTable,
  text,
  timestamp,
  primaryKey
} from "drizzle-orm/pg-core"
import { users } from "./users-schema"

export const subscriptions = pgTable("subscriptions", {
  userId: text("user_id")
    .references(() => users.id)
    .notNull(),
  stripeSubscriptionId: text("stripe_subscription_id")
    .notNull()
    .unique(),
  stripePriceId: text("stripe_price_id").notNull(),
  stripeCurrentPeriodEnd: timestamp("stripe_current_period_end")
    .notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at")
    .defaultNow()
    .notNull()
    .$onUpdate(() => new Date())
}, (table) => ({
  pk: primaryKey(table.userId)
}))

export type Subscription = typeof subscriptions.$inferSelect
export type InsertSubscription = typeof subscriptions.$inferInsert
```

### Subscription Hooks
```typescript
// hooks/useSubscription.ts
'use client'

import { useQuery } from "@tanstack/react-query"
import type { Subscription } from "@/db/schema"

export function useSubscription() {
  return useQuery<Subscription>({
    queryKey: ['subscription'],
    queryFn: async () => {
      const response = await fetch('/api/subscription')
      if (!response.ok) {
        throw new Error('Failed to fetch subscription')
      }
      return response.json()
    }
  })
}

// hooks/useRequireSubscription.ts
'use client'

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useSubscription } from "./useSubscription"

export function useRequireSubscription() {
  const router = useRouter()
  const { data: subscription, isLoading } = useSubscription()
  
  useEffect(() => {
    if (!isLoading && !subscription) {
      router.push('/settings/billing?required=true')
    }
  }, [subscription, isLoading, router])
  
  return {
    hasSubscription: !!subscription,
    isLoading
  }
}
```

### Subscription Components
```typescript
// components/billing/SubscriptionStatus.tsx
'use client'

import { useSubscription } from "@/hooks/useSubscription"
import { ManageSubscriptionButton } from "./ManageSubscriptionButton"

export function SubscriptionStatus() {
  const { data: subscription, isLoading } = useSubscription()
  
  if (isLoading) {
    return <SubscriptionStatusSkeleton />
  }
  
  if (!subscription) {
    return (
      <div className="rounded-lg border p-6">
        <h3 className="text-lg font-medium">No Active Subscription</h3>
        <p className="mt-2 text-gray-500">
          Upgrade to access premium features
        </p>
      </div>
    )
  }
  
  return (
    <div className="rounded-lg border p-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium">Active Subscription</h3>
          <p className="mt-2 text-gray-500">
            Next billing date:{" "}
            {new Date(subscription.stripeCurrentPeriodEnd)
              .toLocaleDateString()}
          </p>
        </div>
        
        <ManageSubscriptionButton />
      </div>
    </div>
  )
}
``` 