# 8. Authentication & Authorization

## 8.1 Clerk Implementation

### Configuration
```typescript
// app/api/auth/[...clerk]/route.ts
import { authMiddleware } from "@clerk/nextjs"

export default authMiddleware({
  // Public routes that don't require authentication
  publicRoutes: [
    "/",
    "/api/health",
    "/api/webhook/stripe"
  ],
  
  // Routes that can be accessed while loading auth
  ignoredRoutes: [
    "/api/webhook/clerk"
  ]
})

export const config = {
  matcher: ["/((?!.*\\..*|_next).*)", "/", "/(api|trpc)(.*)"]
}

// middleware.ts
import { authMiddleware } from "@clerk/nextjs"
import { NextResponse } from "next/server"

export default authMiddleware({
  afterAuth(auth, req) {
    // Handle auth state
    if (!auth.userId && !auth.isPublicRoute) {
      const signInUrl = new URL('/sign-in', req.url)
      signInUrl.searchParams.set('redirect_url', req.url)
      return NextResponse.redirect(signInUrl)
    }

    // Add user context to headers
    if (auth.userId) {
      const requestHeaders = new Headers(req.headers)
      requestHeaders.set('x-user-id', auth.userId)
      return NextResponse.next({
        request: {
          headers: requestHeaders
        }
      })
    }

    return NextResponse.next()
  }
})
```

### Auth Components
```typescript
// components/auth/SignInForm.tsx
'use client'

import { SignIn } from "@clerk/nextjs"
import { dark } from "@clerk/themes"
import { useTheme } from "next-themes"

export function SignInForm() {
  const { theme } = useTheme()
  
  return (
    <SignIn
      appearance={{
        baseTheme: theme === "dark" ? dark : undefined,
        elements: {
          formButtonPrimary: "bg-primary-600 hover:bg-primary-700",
          card: "shadow-none"
        }
      }}
      afterSignInUrl="/dashboard"
      signUpUrl="/sign-up"
    />
  )
}

// components/auth/UserButton.tsx
'use client'

import { UserButton as ClerkUserButton } from "@clerk/nextjs"
import { dark } from "@clerk/themes"
import { useTheme } from "next-themes"

export function UserButton() {
  const { theme } = useTheme()
  
  return (
    <ClerkUserButton
      appearance={{
        baseTheme: theme === "dark" ? dark : undefined,
        elements: {
          avatarBox: "w-8 h-8"
        }
      }}
      afterSignOutUrl="/"
    />
  )
}
```

### Auth Hooks
```typescript
// hooks/useAuth.ts
import { useUser } from "@clerk/nextjs"
import { useQuery } from "@tanstack/react-query"

export function useAuth() {
  const { user, isLoaded, isSignedIn } = useUser()
  
  const { data: permissions } = useQuery({
    queryKey: ['permissions', user?.id],
    queryFn: async () => {
      if (!user?.id) return []
      const response = await fetch('/api/permissions')
      return response.json()
    },
    enabled: !!user?.id
  })
  
  return {
    user,
    isLoaded,
    isSignedIn,
    permissions: permissions || [],
    hasPermission: (permission: string) => 
      permissions?.includes(permission) || false
  }
}

// hooks/useProtectedRoute.ts
import { useAuth } from "./useAuth"
import { useRouter } from "next/navigation"
import { useEffect } from "react"

export function useProtectedRoute(requiredPermissions: string[] = []) {
  const { isLoaded, isSignedIn, hasPermission } = useAuth()
  const router = useRouter()
  
  useEffect(() => {
    if (!isLoaded) return
    
    if (!isSignedIn) {
      router.push('/sign-in')
      return
    }
    
    const hasAllPermissions = requiredPermissions.every(hasPermission)
    if (!hasAllPermissions) {
      router.push('/unauthorized')
    }
  }, [isLoaded, isSignedIn, hasPermission, router, requiredPermissions])
  
  return { isLoaded, isSignedIn }
}
```

## 8.2 Protected Routes

### Route Protection
```typescript
// middleware.ts
import { authMiddleware } from "@clerk/nextjs"
import { NextResponse } from "next/server"

// Define protected routes and their required permissions
const protectedRoutes = {
  '/dashboard': ['view:dashboard'],
  '/strategies': ['manage:strategies'],
  '/settings': ['manage:settings'],
  '/api/strategies': ['manage:strategies'],
  '/api/trades': ['view:trades']
}

export default authMiddleware({
  afterAuth: async (auth, req) => {
    // Check if route requires protection
    const path = new URL(req.url).pathname
    const requiredPermissions = Object.entries(protectedRoutes)
      .find(([route]) => path.startsWith(route))?.[1]
      
    if (!requiredPermissions) {
      return NextResponse.next()
    }
    
    // Redirect to sign in if not authenticated
    if (!auth.userId) {
      const signInUrl = new URL('/sign-in', req.url)
      signInUrl.searchParams.set('redirect_url', req.url)
      return NextResponse.redirect(signInUrl)
    }
    
    // Check permissions
    const hasPermissions = await checkPermissions(
      auth.userId,
      requiredPermissions
    )
    
    if (!hasPermissions) {
      return NextResponse.redirect(new URL('/unauthorized', req.url))
    }
    
    return NextResponse.next()
  }
})

// app/api/permissions/route.ts
import { auth } from "@clerk/nextjs"
import { db } from "@/lib/db"
import { eq } from "drizzle-orm"
import { users, userPermissions } from "@/db/schema"

export async function GET() {
  const { userId } = auth()
  
  if (!userId) {
    return new Response("Unauthorized", { status: 401 })
  }
  
  const permissions = await db.query.userPermissions.findMany({
    where: eq(userPermissions.userId, userId),
    columns: {
      permission: true
    }
  })
  
  return Response.json(permissions.map(p => p.permission))
}
```

### Protected Components
```typescript
// components/ProtectedComponent.tsx
'use client'

import { useAuth } from "@/hooks/useAuth"

interface ProtectedComponentProps {
  children: React.ReactNode
  requiredPermissions?: string[]
  fallback?: React.ReactNode
}

export function ProtectedComponent({
  children,
  requiredPermissions = [],
  fallback = null
}: ProtectedComponentProps) {
  const { isSignedIn, hasPermission } = useAuth()
  
  if (!isSignedIn) {
    return fallback
  }
  
  const hasAllPermissions = requiredPermissions.every(hasPermission)
  if (!hasAllPermissions) {
    return fallback
  }
  
  return <>{children}</>
}

// Usage example
<ProtectedComponent
  requiredPermissions={['manage:strategies']}
  fallback={<UnauthorizedMessage />}
>
  <StrategyControls />
</ProtectedComponent>
```

## 8.3 Session Management

### Session Configuration
```typescript
// lib/session.ts
import { createClerkClient } from "@clerk/nextjs"

const clerk = createClerkClient({
  secretKey: process.env.CLERK_SECRET_KEY
})

export async function getSession(sessionId: string) {
  try {
    const session = await clerk.sessions.getSession(sessionId)
    return session
  } catch (error) {
    console.error('Failed to get session:', error)
    return null
  }
}

export async function revokeSession(sessionId: string) {
  try {
    await clerk.sessions.revokeSession(sessionId)
    return true
  } catch (error) {
    console.error('Failed to revoke session:', error)
    return false
  }
}

export async function getActiveSessions(userId: string) {
  try {
    const sessions = await clerk.users.getSessions(userId)
    return sessions
  } catch (error) {
    console.error('Failed to get active sessions:', error)
    return []
  }
}
```

### Session Management UI
```typescript
// components/settings/SessionManagement.tsx
'use client'

import { useQuery, useMutation } from "@tanstack/react-query"
import { getActiveSessions, revokeSession } from "@/lib/session"
import { useAuth } from "@/hooks/useAuth"

export function SessionManagement() {
  const { user } = useAuth()
  
  const { data: sessions, isLoading } = useQuery({
    queryKey: ['sessions', user?.id],
    queryFn: () => getActiveSessions(user!.id),
    enabled: !!user?.id
  })
  
  const { mutate: handleRevoke } = useMutation({
    mutationFn: revokeSession,
    onSuccess: () => {
      // Refresh sessions list
      queryClient.invalidateQueries(['sessions'])
    }
  })
  
  if (isLoading) {
    return <SessionsSkeleton />
  }
  
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-medium">Active Sessions</h2>
      
      <div className="divide-y divide-gray-200">
        {sessions?.map(session => (
          <div
            key={session.id}
            className="py-4 flex items-center justify-between"
          >
            <div>
              <p className="font-medium">{session.deviceType}</p>
              <p className="text-sm text-gray-500">
                Last active: {formatDate(session.lastActiveAt)}
              </p>
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleRevoke(session.id)}
            >
              Revoke
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}
```

## 8.4 Permission Management

### Permission Types
```typescript
// types/permissions.ts
export type Permission =
  | 'view:dashboard'
  | 'manage:strategies'
  | 'manage:settings'
  | 'view:trades'
  | 'execute:trades'
  | 'manage:users'

export interface Role {
  id: string
  name: string
  permissions: Permission[]
}

export interface UserPermissions {
  userId: string
  permissions: Permission[]
  roles: Role[]
}
```

### Permission Management
```typescript
// lib/permissions.ts
import { db } from "@/lib/db"
import { eq, and } from "drizzle-orm"
import { 
  users,
  roles,
  userRoles,
  permissions,
  rolePermissions
} from "@/db/schema"
import type { Permission, Role } from "@/types"

export async function getUserPermissions(userId: string): Promise<Permission[]> {
  // Get direct permissions
  const directPermissions = await db.query.permissions.findMany({
    where: eq(permissions.userId, userId)
  })
  
  // Get role-based permissions
  const userRolesWithPermissions = await db.query.userRoles.findMany({
    where: eq(userRoles.userId, userId),
    with: {
      role: {
        with: {
          permissions: true
        }
      }
    }
  })
  
  const rolePermissions = userRolesWithPermissions.flatMap(
    ur => ur.role.permissions
  )
  
  // Combine and deduplicate permissions
  const allPermissions = [
    ...directPermissions,
    ...rolePermissions
  ]
  
  return [...new Set(allPermissions)]
}

export async function assignUserRole(
  userId: string,
  roleId: string
): Promise<void> {
  await db.insert(userRoles).values({
    userId,
    roleId
  })
}

export async function removeUserRole(
  userId: string,
  roleId: string
): Promise<void> {
  await db.delete(userRoles).where(
    and(
      eq(userRoles.userId, userId),
      eq(userRoles.roleId, roleId)
    )
  )
}

export async function createRole(
  name: string,
  permissions: Permission[]
): Promise<Role> {
  const [role] = await db.insert(roles)
    .values({ name })
    .returning()
    
  await db.insert(rolePermissions).values(
    permissions.map(permission => ({
      roleId: role.id,
      permission
    }))
  )
  
  return {
    ...role,
    permissions
  }
}
```

### Permission UI
```typescript
// components/settings/PermissionManager.tsx
'use client'

import { useQuery, useMutation } from "@tanstack/react-query"
import { assignUserRole, removeUserRole } from "@/lib/permissions"
import type { Role, UserPermissions } from "@/types"

interface PermissionManagerProps {
  userId: string
}

export function PermissionManager({ userId }: PermissionManagerProps) {
  const { data: userPermissions } = useQuery<UserPermissions>({
    queryKey: ['permissions', userId],
    queryFn: () => fetch(`/api/users/${userId}/permissions`).then(r => r.json())
  })
  
  const { data: availableRoles } = useQuery<Role[]>({
    queryKey: ['roles'],
    queryFn: () => fetch('/api/roles').then(r => r.json())
  })
  
  const assignRole = useMutation({
    mutationFn: (roleId: string) => assignUserRole(userId, roleId),
    onSuccess: () => {
      queryClient.invalidateQueries(['permissions', userId])
    }
  })
  
  const removeRole = useMutation({
    mutationFn: (roleId: string) => removeUserRole(userId, roleId),
    onSuccess: () => {
      queryClient.invalidateQueries(['permissions', userId])
    }
  })
  
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium">Current Permissions</h3>
        <div className="mt-2 space-x-2">
          {userPermissions?.permissions.map(permission => (
            <Badge key={permission}>{permission}</Badge>
          ))}
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-medium">Roles</h3>
        <div className="mt-4 space-y-4">
          {availableRoles?.map(role => {
            const hasRole = userPermissions?.roles.some(
              r => r.id === role.id
            )
            
            return (
              <div
                key={role.id}
                className="flex items-center justify-between"
              >
                <div>
                  <p className="font-medium">{role.name}</p>
                  <p className="text-sm text-gray-500">
                    {role.permissions.join(', ')}
                  </p>
                </div>
                
                <Button
                  variant={hasRole ? 'outline' : 'primary'}
                  onClick={() => {
                    if (hasRole) {
                      removeRole.mutate(role.id)
                    } else {
                      assignRole.mutate(role.id)
                    }
                  }}
                >
                  {hasRole ? 'Remove' : 'Assign'}
                </Button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
``` 