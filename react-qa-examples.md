# React & React Native Examples (3+ Years Experience)

## Core React Concepts

### 1. Virtual DOM Example

```jsx
// Before update
const oldVDOM = {
  type: "div",
  props: { className: "container" },
  children: [{ type: "h1", props: {}, children: ["Hello"] }],
};

// After state change
const newVDOM = {
  type: "div",
  props: { className: "container" },
  children: [{ type: "h1", props: {}, children: ["Hello World"] }],
};

// React will only update the text content, not the entire DOM
```

### 2. Component Lifecycle Example

```jsx
class ExampleComponent extends React.Component {
  componentDidMount() {
    console.log("Component mounted");
    // Fetch data, setup subscriptions
  }

  componentDidUpdate(prevProps, prevState) {
    if (this.props.userID !== prevProps.userID) {
      // Fetch new data when userID changes
    }
  }

  componentWillUnmount() {
    console.log("Component will unmount");
    // Cleanup subscriptions
  }

  render() {
    return <div>Lifecycle Example</div>;
  }
}
```

### 3. React Hooks Example

```jsx
// Before Hooks (Class Component)
class Counter extends React.Component {
  state = { count: 0 };

  increment = () => {
    this.setState({ count: this.state.count + 1 });
  };

  render() {
    return <button onClick={this.increment}>Count: {this.state.count}</button>;
  }
}

// After Hooks (Functional Component)
function Counter() {
  const [count, setCount] = useState(0);

  const increment = () => {
    setCount(count + 1);
  };

  return <button onClick={increment}>Count: {count}</button>;
}
```

### 4. useState, useEffect, useContext Example

```jsx
// Context creation
const ThemeContext = React.createContext("light");

function App() {
  const [theme, setTheme] = useState("light");
  const [count, setCount] = useState(0);

  // useEffect for side effects
  useEffect(() => {
    document.title = `Count: ${count}`;
  }, [count]); // Only re-run when count changes

  return (
    <ThemeContext.Provider value={theme}>
      <button onClick={() => setTheme("dark")}>Toggle Theme</button>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </ThemeContext.Provider>
  );
}
```

### 5. Controlled vs Uncontrolled Components

```jsx
// Controlled Component
function ControlledForm() {
  const [value, setValue] = useState("");

  return <input value={value} onChange={(e) => setValue(e.target.value)} />;
}

// Uncontrolled Component
function UncontrolledForm() {
  const inputRef = useRef();

  const handleSubmit = () => {
    console.log(inputRef.current.value);
  };

  return <input ref={inputRef} />;
}
```

## Performance Optimization

### 6. Performance Optimization Examples

```jsx
// React.memo example
const ExpensiveComponent = React.memo(function ExpensiveComponent({ data }) {
  return <div>{data}</div>;
});

// useMemo example
function Component({ items }) {
  const sortedItems = useMemo(() => {
    return items.sort((a, b) => a - b);
  }, [items]);

  return <List items={sortedItems} />;
}

// useCallback example
function Parent() {
  const [count, setCount] = useState(0);

  const handleClick = useCallback(() => {
    setCount(count + 1);
  }, [count]);

  return <Child onClick={handleClick} />;
}

// Code splitting example
const LazyComponent = React.lazy(() => import("./HeavyComponent"));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <LazyComponent />
    </Suspense>
  );
}
```

### 7. React.memo Example

```jsx
const UserProfile = React.memo(function UserProfile({ user }) {
  return (
    <div>
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  );
});

// Only re-renders when user prop changes
```

### 8. useMemo and useCallback Examples

```jsx
function Calculator({ a, b }) {
  // Memoize expensive calculation
  const result = useMemo(() => {
    return expensiveCalculation(a, b);
  }, [a, b]);

  // Memoize callback function
  const handleClick = useCallback(() => {
    console.log("Clicked!");
  }, []);

  return (
    <div>
      <p>Result: {result}</p>
      <button onClick={handleClick}>Click me</button>
    </div>
  );
}
```

### 9. Code Splitting Example

```jsx
// Route-based code splitting
const Home = React.lazy(() => import("./Home"));
const About = React.lazy(() => import("./About"));

function App() {
  return (
    <Router>
      <Suspense fallback={<div>Loading...</div>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Suspense>
    </Router>
  );
}
```

## State Management

### 10. Redux vs Context API vs React Query

```jsx
// Redux Example
const store = configureStore({
  reducer: {
    counter: counterReducer,
  },
});

function Counter() {
  const count = useSelector((state) => state.counter);
  const dispatch = useDispatch();

  return <button onClick={() => dispatch(increment())}>Count: {count}</button>;
}

// Context API Example
const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("light");

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// React Query Example
function UserProfile() {
  const { data, isLoading } = useQuery("user", fetchUser);

  if (isLoading) return <div>Loading...</div>;

  return <div>{data.name}</div>;
}
```

### 11. Redux Principles Example

```jsx
// Single source of truth
const initialState = {
  user: null,
  settings: {},
};

// State is read-only
function reducer(state = initialState, action) {
  switch (action.type) {
    case "SET_USER":
      return { ...state, user: action.payload };
    default:
      return state;
  }
}

// Changes through pure functions
const increment = () => ({
  type: "INCREMENT",
});

// Unidirectional data flow
function Counter() {
  const count = useSelector((state) => state.count);
  const dispatch = useDispatch();

  return <button onClick={() => dispatch(increment())}>{count}</button>;
}
```

### 12. Redux Middleware Example

```jsx
// Redux Thunk
const fetchUser = (userId) => async (dispatch) => {
  dispatch({ type: "FETCH_USER_START" });
  try {
    const user = await api.getUser(userId);
    dispatch({ type: "FETCH_USER_SUCCESS", payload: user });
  } catch (error) {
    dispatch({ type: "FETCH_USER_ERROR", error });
  }
};

// Redux Saga
function* fetchUserSaga(action) {
  try {
    const user = yield call(api.getUser, action.payload);
    yield put({ type: "FETCH_USER_SUCCESS", payload: user });
  } catch (error) {
    yield put({ type: "FETCH_USER_ERROR", error });
  }
}
```

## React Native Specific

### 13. React vs React Native Example

```jsx
// React (Web)
function WebComponent() {
  return (
    <div className="container">
      <h1>Hello Web</h1>
    </div>
  );
}

// React Native
function NativeComponent() {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Hello Native</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  text: {
    fontSize: 20,
  },
});
```

### 14. React Native Bridge Example

```jsx
// JavaScript
const { NativeModules } = require('react-native');
NativeModules.CustomModule.doSomething();

// Native (Java)
public class CustomModule extends ReactContextBaseJavaModule {
  @ReactMethod
  public void doSomething() {
    // Native implementation
  }
}
```

### 15. React Native Performance Example

```jsx
// Optimized list rendering
import { FlatList } from "react-native";

function OptimizedList() {
  const renderItem = useCallback(({ item }) => <ListItem item={item} />, []);

  return (
    <FlatList
      data={items}
      renderItem={renderItem}
      keyExtractor={(item) => item.id}
      initialNumToRender={10}
      maxToRenderPerBatch={10}
      windowSize={5}
    />
  );
}

// Image caching
import FastImage from "react-native-fast-image";

function CachedImage() {
  return (
    <FastImage
      source={{
        uri: "https://example.com/image.jpg",
        priority: FastImage.priority.normal,
        cache: FastImage.cacheControl.immutable,
      }}
    />
  );
}
```

### 16. React Native Threading Example

```jsx
// JavaScript Thread
function HeavyComputation() {
  const result = useMemo(() => {
    // Heavy computation
    return compute();
  }, []);

  return <Text>{result}</Text>;
}

// Native Module (Runs on Native Thread)
const { HeavyComputationModule } = NativeModules;

function OptimizedComputation() {
  const [result, setResult] = useState(null);

  useEffect(() => {
    HeavyComputationModule.compute().then(setResult);
  }, []);

  return <Text>{result}</Text>;
}
```

## Advanced Concepts

### 17. useEffect vs useLayoutEffect Example

```jsx
function Example() {
  const [width, setWidth] = useState(0);
  const divRef = useRef();

  // useEffect runs after paint
  useEffect(() => {
    console.log("Width after paint:", width);
  }, [width]);

  // useLayoutEffect runs before paint
  useLayoutEffect(() => {
    const newWidth = divRef.current.offsetWidth;
    setWidth(newWidth);
  }, []);

  return <div ref={divRef}>Example</div>;
}
```

### 18. Reconciliation Example

```jsx
// Elements of different types
function Example() {
  return condition ? <div>Hello</div> : <span>World</span>;
}

// Keys in lists
function List() {
  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
}
```

### 19. React Portals Example

```jsx
function Modal() {
  return ReactDOM.createPortal(
    <div className="modal">
      <h1>Modal Content</h1>
    </div>,
    document.getElementById("modal-root")
  );
}
```

### 20. Error Boundaries Example

```jsx
class ErrorBoundary extends React.Component {
  state = { hasError: false };

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    logErrorToService(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }

    return this.props.children;
  }
}

// Usage
<ErrorBoundary>
  <MyComponent />
</ErrorBoundary>;
```

## Testing

### 21. Testing Approaches Example

```jsx
// Unit Test
test("adds 1 + 2 to equal 3", () => {
  expect(sum(1, 2)).toBe(3);
});

// Component Test
test("renders greeting", () => {
  render(<Greeting name="John" />);
  expect(screen.getByText("Hello, John!")).toBeInTheDocument();
});

// Integration Test
test("complete form submission", async () => {
  render(<Form />);
  await userEvent.type(screen.getByLabelText("Name"), "John");
  await userEvent.click(screen.getByText("Submit"));
  expect(await screen.findByText("Success!")).toBeInTheDocument();
});

// E2E Test
describe("App", () => {
  it("should login successfully", () => {
    cy.visit("/login");
    cy.get('[data-testid="email"]').type("user@example.com");
    cy.get('[data-testid="password"]').type("password");
    cy.get('[data-testid="submit"]').click();
    cy.url().should("include", "/dashboard");
  });
});
```

### 22. Component Testing Example

```jsx
// Component
function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}

// Test
test("increments counter", () => {
  render(<Counter />);

  const button = screen.getByText("Increment");
  const count = screen.getByText(/count/i);

  expect(count).toHaveTextContent("Count: 0");

  fireEvent.click(button);

  expect(count).toHaveTextContent("Count: 1");
});
```

## Architecture

### 23. Project Structure Example

```
src/
  ├── components/
  │   ├── common/
  │   └── features/
  ├── hooks/
  ├── services/
  ├── store/
  ├── utils/
  └── types/
```

### 24. Container/Presenter Pattern Example

```jsx
// Container
function UserListContainer() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUsers().then((users) => {
      setUsers(users);
      setLoading(false);
    });
  }, []);

  return <UserList users={users} loading={loading} />;
}

// Presenter
function UserList({ users, loading }) {
  if (loading) return <div>Loading...</div>;

  return (
    <ul>
      {users.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

## Security

### 25. Security Examples

```jsx
// XSS Protection
function SafeComponent({ userInput }) {
  // Don't do this
  return <div dangerouslySetInnerHTML={{ __html: userInput }} />;

  // Do this instead
  return <div>{userInput}</div>;
}

// CSRF Protection
function SecureForm() {
  const [csrfToken, setCsrfToken] = useState("");

  useEffect(() => {
    fetch("/api/csrf-token")
      .then((res) => res.json())
      .then((data) => setCsrfToken(data.token));
  }, []);

  return (
    <form>
      <input type="hidden" name="csrf" value={csrfToken} />
      {/* form fields */}
    </form>
  );
}
```

## Build and Deployment

### 26. Build Optimization Example

```jsx
// webpack.config.js
module.exports = {
  optimization: {
    splitChunks: {
      chunks: "all",
      minSize: 20000,
      maxSize: 0,
      minChunks: 1,
      maxAsyncRequests: 30,
      maxInitialRequests: 30,
      automaticNameDelimiter: "~",
      cacheGroups: {
        vendors: {
          test: /[\\/]node_modules[\\/]/,
          priority: -10,
        },
        default: {
          minChunks: 2,
          priority: -20,
          reuseExistingChunk: true,
        },
      },
    },
  },
};
```

## Additional Questions

### 27. useRef vs useState Example

```jsx
function Example() {
  const [count, setCount] = useState(0);
  const prevCountRef = useRef();

  useEffect(() => {
    prevCountRef.current = count;
  });

  const prevCount = prevCountRef.current;

  return (
    <div>
      <p>Current: {count}</p>
      <p>Previous: {prevCount}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

### 28. Strict Mode Example

```jsx
function App() {
  return (
    <React.StrictMode>
      <div>
        <Component />
      </div>
    </React.StrictMode>
  );
}
```

### 29. Fragments Example

```jsx
function List() {
  return (
    <>
      <li>Item 1</li>
      <li>Item 2</li>
      <li>Item 3</li>
    </>
  );
}
```

### 30. Form Handling Example

```jsx
function Form() {
  const { register, handleSubmit, errors } = useForm();

  const onSubmit = (data) => {
    console.log(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        name="email"
        ref={register({
          required: true,
          pattern: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
        })}
      />
      {errors.email && <span>Invalid email</span>}

      <button type="submit">Submit</button>
    </form>
  );
}
```

### 31. Prop Drilling Solution Example

```jsx
// Before (Prop Drilling)
function App() {
  const [theme, setTheme] = useState("light");
  return <Header theme={theme} />;
}

function Header({ theme }) {
  return <Navigation theme={theme} />;
}

function Navigation({ theme }) {
  return <Button theme={theme} />;
}

// After (Context)
const ThemeContext = createContext();

function App() {
  const [theme, setTheme] = useState("light");
  return (
    <ThemeContext.Provider value={theme}>
      <Header />
    </ThemeContext.Provider>
  );
}

function Header() {
  return <Navigation />;
}

function Navigation() {
  return <Button />;
}

function Button() {
  const theme = useContext(ThemeContext);
  return <button className={theme}>Click me</button>;
}
```

### 32. Synthetic Events Example

```jsx
function Example() {
  const handleClick = (e) => {
    e.preventDefault();
    console.log(e.nativeEvent); // Original browser event
  };

  return <button onClick={handleClick}>Click me</button>;
}
```

### 33. Hooks Rules Example

```jsx
// Correct
function Example() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState("");

  useEffect(() => {
    document.title = `${name} - ${count}`;
  });

  return <div>{count}</div>;
}

// Incorrect
function Example() {
  if (condition) {
    const [count, setCount] = useState(0); // ❌
  }

  useEffect(() => {
    const [name, setName] = useState(""); // ❌
  });

  return <div>{count}</div>;
}
```

### 34. Authentication Example

```jsx
function PrivateRoute({ children }) {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Redirect to="/login" />;
  }

  return children;
}

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/login" component={Login} />
        <PrivateRoute path="/dashboard">
          <Dashboard />
        </PrivateRoute>
      </Switch>
    </Router>
  );
}
```

### 35. SSR Example

```jsx
// Next.js page
export async function getServerSideProps() {
  const data = await fetchData();
  return {
    props: { data },
  };
}

function Page({ data }) {
  return <div>{data}</div>;
}

export default Page;
```

### 36. Concurrent Features Example

```jsx
function App() {
  const [resource, setResource] = useState(initialResource);

  return (
    <Suspense fallback={<h1>Loading...</h1>}>
      <Profile resource={resource} />
    </Suspense>
  );
}

function Profile({ resource }) {
  const user = resource.user.read();
  return <h1>{user.name}</h1>;
}
```

### 37. New Features Example

```jsx
// Automatic batching
function App() {
  const [count, setCount] = useState(0);
  const [flag, setFlag] = useState(false);

  function handleClick() {
    setCount((c) => c + 1); // Does not trigger re-render
    setFlag((f) => !f); // Does not trigger re-render
    // React will batch these updates
  }

  return <button onClick={handleClick}>Next</button>;
}
```

### 38. Internationalization Example

```jsx
import { useTranslation } from "react-i18next";

function Greeting() {
  const { t, i18n } = useTranslation();

  return (
    <div>
      <h1>{t("welcome")}</h1>
      <button onClick={() => i18n.changeLanguage("en")}>English</button>
      <button onClick={() => i18n.changeLanguage("fr")}>Français</button>
    </div>
  );
}
```

### 39. Design Patterns Example

```jsx
// Higher-Order Component
function withLoading(Component) {
  return function WithLoading({ isLoading, ...props }) {
    if (isLoading) return <div>Loading...</div>;
    return <Component {...props} />;
  };
}

// Render Props
function MouseTracker() {
  return (
    <Mouse
      render={(mouse) => (
        <p>
          The mouse position is {mouse.x}, {mouse.y}
        </p>
      )}
    />
  );
}

// Compound Components
function Tabs({ children }) {
  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <div>
      {React.Children.map(children, (child, index) =>
        React.cloneElement(child, {
          isActive: index === activeIndex,
          onSelect: () => setActiveIndex(index),
        })
      )}
    </div>
  );
}
```

### 40. Routing Example

```jsx
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route
          path="/dashboard"
          element={
            <RequireAuth>
              <Dashboard />
            </RequireAuth>
          }
        />
      </Routes>
    </Router>
  );
}
```

### 41. React vs Next.js Example

```jsx
// React (Client-side only)
function App() {
  return <div>Hello World</div>;
}

// Next.js (Server-side rendering)
export async function getServerSideProps() {
  const data = await fetchData();
  return { props: { data } };
}

function Page({ data }) {
  return <div>{data}</div>;
}

export default Page;
```

### 42. State Persistence Example

```jsx
function usePersistedState(key, defaultValue) {
  const [state, setState] = useState(() => {
    const persisted = localStorage.getItem(key);
    return persisted ? JSON.parse(persisted) : defaultValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);

  return [state, setState];
}

function App() {
  const [theme, setTheme] = usePersistedState("theme", "light");

  return (
    <div className={theme}>
      <button onClick={() => setTheme("dark")}>Toggle Theme</button>
    </div>
  );
}
```

### 43. Accessibility Example

```jsx
function AccessibleButton() {
  return (
    <button
      aria-label="Close dialog"
      onClick={handleClose}
      onKeyDown={handleKeyDown}
    >
      <span aria-hidden="true">&times;</span>
    </button>
  );
}

function AccessibleForm() {
  return (
    <form>
      <label htmlFor="email">Email:</label>
      <input
        id="email"
        type="email"
        aria-required="true"
        aria-describedby="email-error"
      />
      <span id="email-error" role="alert">
        Please enter a valid email
      </span>
    </form>
  );
}
```

### 44. Animations Example

```jsx
// CSS Transitions
const Box = styled.div`
  transition: transform 0.3s ease;
  &:hover {
    transform: scale(1.1);
  }
`;

// React Spring
function AnimatedBox() {
  const props = useSpring({
    from: { opacity: 0 },
    to: { opacity: 1 },
  });

  return <animated.div style={props}>Hello</animated.div>;
}

// Framer Motion
function MotionBox() {
  return (
    <motion.div animate={{ x: 100 }} transition={{ duration: 0.5 }}>
      Hello
    </motion.div>
  );
}
```

### 45. Performance Monitoring Example

```jsx
// React DevTools Profiler
function ProfiledComponent() {
  return (
    <Profiler
      id="Navigation"
      onRender={(id, phase, actualDuration) => {
        console.log(`${id}'s ${phase} phase:`, actualDuration);
      }}
    >
      <Navigation />
    </Profiler>
  );
}

// Performance API
function measurePerformance() {
  performance.mark("start");

  // Expensive operation
  expensiveOperation();

  performance.mark("end");
  performance.measure("operation", "start", "end");

  const measures = performance.getEntriesByType("measure");
  console.log(measures[0].duration);
}
```

### 46. Data Fetching Example

```jsx
// Fetch API
function FetchExample() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("https://api.example.com/data")
      .then((res) => res.json())
      .then(setData);
  }, []);

  return <div>{data}</div>;
}

// React Query
function QueryExample() {
  const { data, isLoading } = useQuery("todos", fetchTodos);

  if (isLoading) return <div>Loading...</div>;

  return (
    <ul>
      {data.map((todo) => (
        <li key={todo.id}>{todo.title}</li>
      ))}
    </ul>
  );
}

// Custom Hook
function useData(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      });
  }, [url]);

  return { data, loading };
}
```

### 47. Testing Best Practices Example

```jsx
// Component
function LoginForm({ onSubmit }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ email, password });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        data-testid="email"
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        data-testid="password"
      />
      <button type="submit">Login</button>
    </form>
  );
}

// Test
test("submits form with email and password", async () => {
  const handleSubmit = jest.fn();
  render(<LoginForm onSubmit={handleSubmit} />);

  await userEvent.type(screen.getByTestId("email"), "test@example.com");
  await userEvent.type(screen.getByTestId("password"), "password123");
  await userEvent.click(screen.getByText("Login"));

  expect(handleSubmit).toHaveBeenCalledWith({
    email: "test@example.com",
    password: "password123",
  });
});
```

### 48. Error Boundaries Example

```jsx
class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    logErrorToService(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div>
          <h1>Something went wrong</h1>
          <p>{this.state.error.message}</p>
          <button onClick={() => this.setState({ hasError: false })}>
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Usage
<ErrorBoundary>
  <BuggyComponent />
</ErrorBoundary>;
```

### 49. State Management Patterns Example

```jsx
// Local State
function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}

// Context API
const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("light");
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// Redux
const store = configureStore({
  reducer: {
    counter: counterReducer,
  },
});

function Counter() {
  const count = useSelector((state) => state.counter);
  const dispatch = useDispatch();
  return <button onClick={() => dispatch(increment())}>{count}</button>;
}
```

### 50. Styling Example

```jsx
// CSS Modules
import styles from "./Button.module.css";

function Button() {
  return <button className={styles.button}>Click me</button>;
}

// Styled Components
const Button = styled.button`
  background: ${(props) => (props.primary ? "blue" : "white")};
  color: ${(props) => (props.primary ? "white" : "blue")};
`;

// CSS-in-JS
const styles = {
  button: {
    backgroundColor: "blue",
    color: "white",
    padding: "10px",
  },
};

function Button() {
  return <button style={styles.button}>Click me</button>;
}

// Utility-first CSS (Tailwind)
function Button() {
  return (
    <button className="bg-blue-500 text-white px-4 py-2 rounded">
      Click me
    </button>
  );
}
```
