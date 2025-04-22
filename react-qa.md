# React & React Native Interview Questions (3+ Years Experience)

## Core React Concepts

### 1. What is the Virtual DOM and how does it work?

The Virtual DOM is a lightweight copy of the actual DOM. React uses it to improve performance by:

- Creating a virtual representation of the UI in memory
- Comparing it with the previous version (diffing)
- Updating only the changed parts in the real DOM
- Batching multiple updates together

### 2. Explain React's component lifecycle methods

Class components have these main lifecycle methods:

- `componentDidMount`: After first render
- `componentDidUpdate`: After updates
- `componentWillUnmount`: Before component removal
- `shouldComponentUpdate`: Controls re-rendering
- `getDerivedStateFromProps`: Updates state based on props
- `getSnapshotBeforeUpdate`: Captures DOM info before update

### 3. What are React Hooks and why were they introduced?

Hooks are functions that let you use state and other React features in functional components. They were introduced to:

- Solve the problem of "wrapper hell"
- Make code more reusable
- Simplify complex components
- Make stateful logic shareable

### 4. Explain useState, useEffect, and useContext

- `useState`: Manages local component state
- `useEffect`: Handles side effects (data fetching, subscriptions)
- `useContext`: Accesses context values without prop drilling

### 5. What is the difference between controlled and uncontrolled components?

- Controlled: Form data is handled by React state
- Uncontrolled: Form data is handled by the DOM itself

## Performance Optimization

### 6. How do you optimize React application performance?

- Use React.memo for component memoization
- Implement useMemo for expensive calculations
- Use useCallback for function references
- Implement code splitting with React.lazy
- Use windowing for large lists
- Optimize re-renders with shouldComponentUpdate

### 7. What is React.memo and when should you use it?

React.memo is a higher-order component that memoizes the rendered output of a component. Use it when:

- Component renders often with the same props
- Component is expensive to render
- Props are primitive values

### 8. Explain useMemo and useCallback

- `useMemo`: Memoizes the result of a function
- `useCallback`: Memoizes a function itself

### 9. What is code splitting and how do you implement it?

Code splitting is dividing your bundle into smaller chunks. Implement using:

- React.lazy
- Dynamic imports
- Route-based code splitting

## State Management

### 10. Compare Redux, Context API, and React Query

- Redux: Global state management with middleware
- Context API: Built-in solution for prop drilling
- React Query: Server state management

### 11. What are the principles of Redux?

- Single source of truth
- State is read-only
- Changes through pure functions
- Unidirectional data flow

### 12. Explain Redux middleware

Middleware provides a third-party extension point between dispatching an action and the moment it reaches the reducer. Examples:

- Redux Thunk
- Redux Saga
- Redux Observable

## React Native Specific

### 13. What is the difference between React and React Native?

- React: Web UI library
- React Native: Mobile app framework
- Different rendering engines
- Different styling approaches
- Different navigation solutions

### 14. How does React Native bridge work?

The bridge is a communication layer between:

- JavaScript thread
- Native thread
- Shadow thread (layout)
- UI thread

### 15. What are the main performance considerations in React Native?

- Minimize bridge communication
- Use native modules when possible
- Optimize list rendering
- Implement proper image caching
- Use Hermes engine

### 16. Explain React Native's threading model

- JavaScript Thread: Runs your React code
- Native Thread: Handles native modules
- Shadow Thread: Calculates layouts
- UI Thread: Renders the actual UI

## Advanced Concepts

### 17. What is the difference between useEffect and useLayoutEffect?

- `useEffect`: Runs after render is committed to screen
- `useLayoutEffect`: Runs before browser paint

### 18. Explain React's reconciliation process

Reconciliation is how React updates the DOM:

- Elements of different types
- DOM elements of the same type
- Component elements of the same type
- Keys in lists

### 19. What are React Portals?

Portals provide a way to render children into a DOM node that exists outside the DOM hierarchy of the parent component.

### 20. Explain React's error boundaries

Error boundaries are React components that:

- Catch JavaScript errors
- Display fallback UI
- Log errors for reporting

## Testing

### 21. What are the main testing approaches in React?

- Unit testing (Jest)
- Component testing (React Testing Library)
- Integration testing
- E2E testing (Cypress)

### 22. How do you test React components?

- Render components in isolation
- Test user interactions
- Verify component output
- Mock external dependencies

## Architecture

### 23. What are the best practices for React project structure?

- Feature-based organization
- Shared components
- Hooks and utilities
- Constants and types
- API services

### 24. Explain the container/presenter pattern

- Container: Handles logic and data
- Presenter: Handles rendering
- Separation of concerns

## Security

### 25. What are common security concerns in React applications?

- XSS attacks
- CSRF protection
- Secure data handling
- Authentication
- Authorization

## Build and Deployment

### 26. How do you optimize React build size?

- Code splitting
- Tree shaking
- Compression
- Image optimization
- Lazy loading

## Additional Questions

### 27. What is the difference between useRef and useState?

- `useRef`: Mutable value that persists between renders
- `useState`: Immutable state that triggers re-renders

### 28. Explain React's strict mode

Strict mode is a tool for:

- Identifying potential problems
- Highlighting deprecated features
- Detecting unexpected side effects

### 29. What are React fragments?

Fragments let you group children without adding extra nodes to the DOM.

### 30. How do you handle forms in React?

- Controlled components
- Form libraries (Formik, React Hook Form)
- Validation
- Error handling

### 31. What is prop drilling and how do you avoid it?

Prop drilling is passing props through multiple levels. Avoid using:

- Context API
- State management
- Component composition

### 32. Explain React's synthetic events

Synthetic events are React's cross-browser wrapper around native events.

### 33. What are React hooks rules?

- Only call hooks at the top level
- Only call hooks from React functions
- Follow the hooks naming convention

### 34. How do you handle authentication in React?

- JWT tokens
- OAuth
- Session management
- Protected routes

### 35. What is server-side rendering in React?

SSR renders React components on the server and sends HTML to the client.

### 36. Explain React's concurrent features

- Automatic batching
- Transitions
- Suspense
- Streaming server rendering

### 37. What are React's new features in recent versions?

- Automatic batching
- New root API
- Strict mode improvements
- Concurrent features

### 38. How do you handle internationalization in React?

- i18next
- React Intl
- Format.js
- Custom solutions

### 39. What are React's design patterns?

- Higher-order components
- Render props
- Compound components
- Controlled components

### 40. How do you handle routing in React?

- React Router
- Route configuration
- Navigation guards
- Lazy loading routes

### 41. What is the difference between React and Next.js?

- React: UI library
- Next.js: React framework
- Server-side rendering
- Static site generation
- API routes

### 42. How do you handle state persistence in React?

- Local storage
- Session storage
- Redux persist
- Custom solutions

### 43. What are React's best practices for accessibility?

- Semantic HTML
- ARIA attributes
- Keyboard navigation
- Screen reader support

### 44. How do you handle animations in React?

- CSS transitions
- React Spring
- Framer Motion
- CSS-in-JS solutions

### 45. What are React's performance monitoring tools?

- React DevTools
- Performance API
- Lighthouse
- Custom solutions

### 46. How do you handle data fetching in React?

- Fetch API
- Axios
- React Query
- SWR
- Custom hooks

### 47. What are React's testing best practices?

- Test behavior, not implementation
- Use testing library
- Mock external dependencies
- Test edge cases

### 48. How do you handle error boundaries in React?

- Try-catch blocks
- Error boundary components
- Error logging
- Fallback UI

### 49. What are React's state management patterns?

- Local state
- Context API
- Redux
- MobX
- Custom solutions

### 50. How do you handle styling in React?

- CSS modules
- Styled-components
- CSS-in-JS
- Utility-first CSS
- Component libraries
