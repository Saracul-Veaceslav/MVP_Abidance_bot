# 12. Testing

## 12.1 Unit Testing

### Test Setup
```typescript
// tests/setup.ts
import "@testing-library/jest-dom"
import { afterEach } from "vitest"
import { cleanup } from "@testing-library/react"
import { mockPostHog } from "@/tests/mocks/posthog"
import { mockStripe } from "@/tests/mocks/stripe"

// Clean up after each test
afterEach(() => {
  cleanup()
  mockPostHog.mockClear()
  mockStripe.mockClear()
})

// Mock environment variables
process.env = {
  ...process.env,
  NEXT_PUBLIC_APP_URL: "http://localhost:3000",
  NEXT_PUBLIC_POSTHOG_KEY: "test_key",
  STRIPE_SECRET_KEY: "test_key"
}
```

### Component Tests
```typescript
// tests/components/strategies/StrategyCard.test.tsx
import { render, screen, fireEvent } from "@testing-library/react"
import { vi } from "vitest"
import { StrategyCard } from "@/components/strategies/StrategyCard"
import { useAction } from "@/hooks/useAction"

/**
 * Feature: Strategy Card Component
 * 
 * Scenario: Displaying and toggling a strategy
 *   Given a strategy card component
 *   When the user views the strategy details
 *   And toggles the strategy activation
 *   Then the strategy details should be visible
 *   And the toggle action should be triggered
 */

vi.mock("@/hooks/useAction")

const mockStrategy = {
  id: "test-id",
  name: "Test Strategy",
  description: "Test Description",
  isActive: false
}

describe("StrategyCard", () => {
  const mockExecute = vi.fn()
  
  beforeEach(() => {
    vi.mocked(useAction).mockReturnValue({
      execute: mockExecute,
      isPending: false,
      error: null
    })
  })
  
  it("displays strategy details", () => {
    render(<StrategyCard strategy={mockStrategy} />)
    
    expect(screen.getByText(mockStrategy.name)).toBeInTheDocument()
    expect(screen.getByText(mockStrategy.description)).toBeInTheDocument()
  })
  
  it("handles strategy toggle", async () => {
    mockExecute.mockResolvedValueOnce({
      isSuccess: true,
      data: { ...mockStrategy, isActive: true }
    })
    
    render(<StrategyCard strategy={mockStrategy} />)
    
    const toggle = screen.getByRole("switch")
    await fireEvent.click(toggle)
    
    expect(mockExecute).toHaveBeenCalledWith(
      expect.any(Function)
    )
  })
  
  it("shows loading state while toggling", () => {
    vi.mocked(useAction).mockReturnValue({
      execute: mockExecute,
      isPending: true,
      error: null
    })
    
    render(<StrategyCard strategy={mockStrategy} />)
    
    expect(screen.getByRole("switch")).toBeDisabled()
  })
})
```

### Hook Tests
```typescript
// tests/hooks/useAction.test.ts
import { renderHook, act } from "@testing-library/react"
import { useAction } from "@/hooks/useAction"

/**
 * Feature: Action Hook
 * 
 * Scenario: Executing an action with success
 *   Given an action hook
 *   When an action is executed
 *   And the action succeeds
 *   Then the success state should be returned
 *   And loading states should be managed correctly
 */

describe("useAction", () => {
  it("handles successful action execution", async () => {
    const mockAction = vi.fn().mockResolvedValue({
      isSuccess: true,
      message: "Success",
      data: { id: 1 }
    })
    
    const { result } = renderHook(() => useAction())
    
    expect(result.current.isPending).toBe(false)
    expect(result.current.error).toBeNull()
    
    let actionResult
    await act(async () => {
      actionResult = await result.current.execute(mockAction)
    })
    
    expect(actionResult).toEqual({
      isSuccess: true,
      message: "Success",
      data: { id: 1 }
    })
    expect(result.current.isPending).toBe(false)
    expect(result.current.error).toBeNull()
  })
  
  it("handles failed action execution", async () => {
    const mockAction = vi.fn().mockResolvedValue({
      isSuccess: false,
      message: "Error"
    })
    
    const { result } = renderHook(() => useAction())
    
    let actionResult
    await act(async () => {
      actionResult = await result.current.execute(mockAction)
    })
    
    expect(actionResult).toEqual({
      isSuccess: false,
      message: "Error"
    })
    expect(result.current.error).toBe("Error")
  })
})
```

### Action Tests
```typescript
// tests/actions/strategies.test.ts
import { vi } from "vitest"
import { createStrategyAction } from "@/actions/strategies"
import { db } from "@/lib/db"
import { auth } from "@clerk/nextjs"

/**
 * Feature: Strategy Actions
 * 
 * Scenario: Creating a new strategy
 *   Given an authenticated user
 *   When they create a new strategy
 *   Then the strategy should be saved to the database
 *   And the cache should be revalidated
 */

vi.mock("@clerk/nextjs")
vi.mock("@/lib/db")

describe("createStrategyAction", () => {
  const mockStrategy = {
    name: "Test Strategy",
    symbol: "BTC/USD",
    timeframe: "1h"
  }
  
  beforeEach(() => {
    vi.mocked(auth).mockReturnValue({
      userId: "test-user"
    } as any)
    
    vi.mocked(db.insert).mockResolvedValue([{
      id: "test-id",
      ...mockStrategy,
      userId: "test-user",
      createdAt: new Date(),
      updatedAt: new Date()
    }])
  })
  
  it("creates a strategy successfully", async () => {
    const result = await createStrategyAction(mockStrategy)
    
    expect(result.isSuccess).toBe(true)
    expect(result.data).toMatchObject({
      id: "test-id",
      ...mockStrategy
    })
    expect(db.insert).toHaveBeenCalledWith(
      expect.any(Object),
      expect.objectContaining(mockStrategy)
    )
  })
  
  it("handles unauthenticated users", async () => {
    vi.mocked(auth).mockReturnValue({} as any)
    
    const result = await createStrategyAction(mockStrategy)
    
    expect(result.isSuccess).toBe(false)
    expect(result.message).toBe("Not authenticated")
    expect(db.insert).not.toHaveBeenCalled()
  })
  
  it("handles database errors", async () => {
    vi.mocked(db.insert).mockRejectedValue(
      new Error("Database error")
    )
    
    const result = await createStrategyAction(mockStrategy)
    
    expect(result.isSuccess).toBe(false)
    expect(result.message).toBe("Failed to create strategy")
  })
})
```

## 12.2 Integration Testing

### API Tests
```typescript
// tests/integration/api/strategies.test.ts
import { createMocks } from "node-mocks-http"
import { POST } from "@/app/api/strategies/route"
import { db } from "@/lib/db"
import { auth } from "@clerk/nextjs"

/**
 * Feature: Strategies API
 * 
 * Scenario: Creating a strategy via API
 *   Given an authenticated request
 *   When a POST request is made to create a strategy
 *   Then the strategy should be created
 *   And the response should contain the strategy data
 */

describe("/api/strategies", () => {
  const mockStrategy = {
    name: "Test Strategy",
    symbol: "BTC/USD",
    timeframe: "1h"
  }
  
  beforeEach(() => {
    vi.mocked(auth).mockReturnValue({
      userId: "test-user"
    } as any)
  })
  
  it("creates a strategy", async () => {
    const { req, res } = createMocks({
      method: "POST",
      body: mockStrategy
    })
    
    vi.mocked(db.insert).mockResolvedValue([{
      id: "test-id",
      ...mockStrategy,
      userId: "test-user",
      createdAt: new Date(),
      updatedAt: new Date()
    }])
    
    const response = await POST(req)
    const data = await response.json()
    
    expect(response.status).toBe(200)
    expect(data).toMatchObject({
      id: "test-id",
      ...mockStrategy
    })
  })
  
  it("validates request body", async () => {
    const { req, res } = createMocks({
      method: "POST",
      body: {}
    })
    
    const response = await POST(req)
    
    expect(response.status).toBe(400)
  })
})
```

### Database Tests
```typescript
// tests/integration/db/strategies.test.ts
import { db } from "@/lib/db"
import { strategiesTable } from "@/db/schema"
import { eq } from "drizzle-orm"

/**
 * Feature: Strategy Database Operations
 * 
 * Scenario: Managing strategies in the database
 *   Given a database connection
 *   When performing CRUD operations
 *   Then the operations should be successful
 *   And the data should be consistent
 */

describe("Strategy Database Operations", () => {
  beforeEach(async () => {
    await db.delete(strategiesTable)
  })
  
  it("creates and retrieves a strategy", async () => {
    const strategy = {
      name: "Test Strategy",
      symbol: "BTC/USD",
      timeframe: "1h",
      userId: "test-user"
    }
    
    const [created] = await db
      .insert(strategiesTable)
      .values(strategy)
      .returning()
    
    expect(created).toMatchObject(strategy)
    
    const retrieved = await db.query.strategies.findFirst({
      where: eq(strategiesTable.id, created.id)
    })
    
    expect(retrieved).toMatchObject(strategy)
  })
  
  it("updates a strategy", async () => {
    const [strategy] = await db
      .insert(strategiesTable)
      .values({
        name: "Test Strategy",
        symbol: "BTC/USD",
        timeframe: "1h",
        userId: "test-user"
      })
      .returning()
    
    const [updated] = await db
      .update(strategiesTable)
      .set({ name: "Updated Strategy" })
      .where(eq(strategiesTable.id, strategy.id))
      .returning()
    
    expect(updated.name).toBe("Updated Strategy")
  })
})
```

## 12.3 End-to-End Testing

### Cypress Setup
```typescript
// cypress/support/commands.ts
import { auth } from "@clerk/nextjs"

Cypress.Commands.add("login", () => {
  cy.intercept("/api/auth/**", (req) => {
    req.headers["Authorization"] = `Bearer test_token`
  })
  
  cy.window().then((win) => {
    win.localStorage.setItem("clerk-user", JSON.stringify({
      id: "test-user",
      email: "test@example.com"
    }))
  })
})

Cypress.Commands.add("createStrategy", (strategy) => {
  cy.request({
    method: "POST",
    url: "/api/strategies",
    body: strategy,
    headers: {
      Authorization: `Bearer test_token`
    }
  })
})
```

### Feature Tests
```typescript
// cypress/e2e/strategies.cy.ts
describe("Strategy Management", () => {
  beforeEach(() => {
    cy.login()
    cy.visit("/strategies")
  })
  
  it("creates a new strategy", () => {
    cy.get("[data-testid=create-strategy-button]").click()
    
    cy.get("[data-testid=strategy-name-input]")
      .type("Test Strategy")
    
    cy.get("[data-testid=strategy-symbol-input]")
      .type("BTC/USD")
    
    cy.get("[data-testid=strategy-timeframe-select]")
      .click()
    cy.get("[data-testid=timeframe-option-1h]")
      .click()
    
    cy.get("[data-testid=create-strategy-submit]")
      .click()
    
    cy.get("[data-testid=strategy-card]")
      .should("contain", "Test Strategy")
      .and("contain", "BTC/USD")
  })
  
  it("toggles strategy activation", () => {
    cy.createStrategy({
      name: "Test Strategy",
      symbol: "BTC/USD",
      timeframe: "1h"
    })
    
    cy.visit("/strategies")
    
    cy.get("[data-testid=strategy-toggle]")
      .click()
    
    cy.get("[data-testid=strategy-status]")
      .should("contain", "Active")
  })
})

// cypress/e2e/trading.cy.ts
describe("Trading Flow", () => {
  beforeEach(() => {
    cy.login()
    cy.visit("/trading")
  })
  
  it("executes a trade", () => {
    cy.get("[data-testid=symbol-input]")
      .type("BTC/USD")
    
    cy.get("[data-testid=amount-input]")
      .type("0.1")
    
    cy.get("[data-testid=buy-button]")
      .click()
    
    cy.get("[data-testid=confirmation-dialog]")
      .should("be.visible")
    
    cy.get("[data-testid=confirm-button]")
      .click()
    
    cy.get("[data-testid=success-message]")
      .should("be.visible")
    
    cy.get("[data-testid=positions-table]")
      .should("contain", "BTC/USD")
      .and("contain", "0.1")
  })
})
```

## 12.4 Performance Testing

### Load Testing
```typescript
// tests/performance/load.test.ts
import { test } from "@playwright/test"
import { chromium } from "playwright"

test.describe("Load Testing", () => {
  test("handles multiple strategy updates", async () => {
    const browser = await chromium.launch()
    const context = await browser.newContext()
    const page = await context.newPage()
    
    await page.goto("/strategies")
    
    const startTime = Date.now()
    
    // Simulate 100 rapid strategy updates
    for (let i = 0; i < 100; i++) {
      await page.click(`[data-testid=strategy-toggle-${i}]`)
    }
    
    const endTime = Date.now()
    const duration = endTime - startTime
    
    expect(duration).toBeLessThan(5000) // Should complete in under 5s
    
    await browser.close()
  })
})
```

### Memory Testing
```typescript
// tests/performance/memory.test.ts
import { test, expect } from "@playwright/test"

test.describe("Memory Usage", () => {
  test("monitors memory during trading operations", async () => {
    const browser = await chromium.launch()
    const context = await browser.newContext()
    const page = await context.newPage()
    
    await page.goto("/trading")
    
    const initialMemory = await page.evaluate(() => 
      performance.memory.usedJSHeapSize
    )
    
    // Simulate trading operations
    for (let i = 0; i < 50; i++) {
      await page.click("[data-testid=execute-trade]")
      await page.waitForTimeout(100)
    }
    
    const finalMemory = await page.evaluate(() => 
      performance.memory.usedJSHeapSize
    )
    
    const memoryIncrease = finalMemory - initialMemory
    expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024) // < 50MB
    
    await browser.close()
  })
})
``` 