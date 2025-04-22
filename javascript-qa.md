# JavaScript Interview Questions (Beginner to Advanced)

## Beginner Level

### 1. What are the different data types in JavaScript?

```javascript
// Primitive Types
const string = "Hello"; // string
const number = 42; // number
const boolean = true; // boolean
const nullValue = null; // null
const undefinedValue = undefined; // undefined
const symbol = Symbol("unique"); // symbol
const bigInt = 9007199254740991n; // bigint

// Reference Types
const object = { name: "John" }; // object
const array = [1, 2, 3]; // array
const functionValue = function () {}; // function
```

### 2. What is the difference between == and ===?

```javascript
// == (loose equality)
console.log(1 == "1"); // true (type coercion)
console.log(0 == false); // true
console.log(null == undefined); // true

// === (strict equality)
console.log(1 === "1"); // false
console.log(0 === false); // false
console.log(null === undefined); // false
```

### 3. What are var, let, and const?

```javascript
// var (function-scoped, hoisted)
var x = 1;
if (true) {
  var x = 2;
}
console.log(x); // 2

// let (block-scoped)
let y = 1;
if (true) {
  let y = 2;
}
console.log(y); // 1

// const (block-scoped, immutable)
const z = 1;
// z = 2; // Error: Assignment to constant variable
```

### 4. What is hoisting?

```javascript
// Function declarations are hoisted
console.log(sayHello()); // "Hello"
function sayHello() {
  return "Hello";
}

// Variable declarations are hoisted but not assignments
console.log(x); // undefined
var x = 5;
```

### 5. What are closures?

```javascript
function outer() {
  const message = "Hello";

  function inner() {
    console.log(message);
  }

  return inner;
}

const innerFunc = outer();
innerFunc(); // "Hello"
```

## Intermediate Level

### 6. What is the 'this' keyword?

```javascript
// Global context
console.log(this); // Window (in browser)

// Object method
const obj = {
  name: "John",
  sayName() {
    console.log(this.name);
  },
};
obj.sayName(); // "John"

// Constructor
function Person(name) {
  this.name = name;
}
const person = new Person("John");
console.log(person.name); // "John"
```

### 7. What are promises?

```javascript
const promise = new Promise((resolve, reject) => {
  setTimeout(() => {
    resolve("Success!");
  }, 1000);
});

promise
  .then((result) => console.log(result)) // "Success!"
  .catch((error) => console.error(error));
```

### 8. What is async/await?

```javascript
async function fetchData() {
  try {
    const response = await fetch("https://api.example.com/data");
    const data = await response.json();
    console.log(data);
  } catch (error) {
    console.error("Error:", error);
  }
}
```

### 9. What is the event loop?

```javascript
console.log("Start");

setTimeout(() => {
  console.log("Timeout");
}, 0);

Promise.resolve().then(() => {
  console.log("Promise");
});

console.log("End");

// Output:
// Start
// End
// Promise
// Timeout
```

### 10. What are arrow functions?

```javascript
// Regular function
function add(a, b) {
  return a + b;
}

// Arrow function
const add = (a, b) => a + b;

// Arrow function with this binding
const obj = {
  name: "John",
  regular: function () {
    console.log(this.name); // "John"
  },
  arrow: () => {
    console.log(this.name); // undefined
  },
};
```

## Advanced Level

### 11. What is prototypal inheritance?

```javascript
function Animal(name) {
  this.name = name;
}

Animal.prototype.speak = function () {
  console.log(`${this.name} makes a sound`);
};

function Dog(name) {
  Animal.call(this, name);
}

Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;

Dog.prototype.speak = function () {
  console.log(`${this.name} barks`);
};

const dog = new Dog("Rex");
dog.speak(); // "Rex barks"
```

### 12. What are generators?

```javascript
function* numberGenerator() {
  yield 1;
  yield 2;
  yield 3;
}

const gen = numberGenerator();
console.log(gen.next().value); // 1
console.log(gen.next().value); // 2
console.log(gen.next().value); // 3
```

### 13. What are proxy objects?

```javascript
const handler = {
  get: function (target, prop) {
    return prop in target ? target[prop] : 37;
  },
};

const p = new Proxy({}, handler);
p.a = 1;
console.log(p.a); // 1
console.log(p.b); // 37
```

### 14. What are decorators?

```javascript
function readonly(target, name, descriptor) {
  descriptor.writable = false;
  return descriptor;
}

class Person {
  @readonly
  name = "John";
}

const person = new Person();
person.name = "Jane"; // Error: Cannot assign to read only property
```

### 15. What is currying?

```javascript
const multiply = (a) => (b) => a * b;
const multiplyByTwo = multiply(2);
console.log(multiplyByTwo(4)); // 8
```

## ES6+ Features

### 16. What are template literals?

```javascript
const name = "John";
const age = 30;
console.log(`My name is ${name} and I am ${age} years old`);
```

### 17. What is destructuring?

```javascript
// Object destructuring
const { name, age } = { name: "John", age: 30 };
console.log(name, age); // "John" 30

// Array destructuring
const [first, second] = [1, 2];
console.log(first, second); // 1 2
```

### 18. What is the spread operator?

```javascript
// Array spreading
const arr1 = [1, 2, 3];
const arr2 = [...arr1, 4, 5];
console.log(arr2); // [1, 2, 3, 4, 5]

// Object spreading
const obj1 = { a: 1, b: 2 };
const obj2 = { ...obj1, c: 3 };
console.log(obj2); // { a: 1, b: 2, c: 3 }
```

### 19. What are default parameters?

```javascript
function greet(name = "Guest") {
  console.log(`Hello, ${name}!`);
}

greet(); // "Hello, Guest!"
greet("John"); // "Hello, John!"
```

### 20. What are rest parameters?

```javascript
function sum(...numbers) {
  return numbers.reduce((a, b) => a + b);
}

console.log(sum(1, 2, 3, 4)); // 10
```

## Error Handling

### 21. What is try-catch?

```javascript
try {
  throw new Error("Something went wrong");
} catch (error) {
  console.error(error.message); // "Something went wrong"
} finally {
  console.log("Cleanup");
}
```

### 22. What are custom errors?

```javascript
class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "ValidationError";
  }
}

try {
  throw new ValidationError("Invalid input");
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(error.message);
  }
}
```

## Functional Programming

### 23. What are pure functions?

```javascript
// Pure function
function add(a, b) {
  return a + b;
}

// Impure function
let counter = 0;
function increment() {
  counter++;
  return counter;
}
```

### 24. What is function composition?

```javascript
const compose = (f, g) => (x) => f(g(x));
const add1 = (x) => x + 1;
const multiply2 = (x) => x * 2;
const addThenMultiply = compose(multiply2, add1);
console.log(addThenMultiply(5)); // 12
```

### 25. What is memoization?

```javascript
function memoize(fn) {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

const fibonacci = memoize((n) => {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
});
```

## Asynchronous Programming

### 26. What is callback hell?

```javascript
// Callback hell
getUser(userId, function (user) {
  getPosts(user.id, function (posts) {
    getComments(posts[0].id, function (comments) {
      console.log(comments);
    });
  });
});

// Using promises
getUser(userId)
  .then((user) => getPosts(user.id))
  .then((posts) => getComments(posts[0].id))
  .then((comments) => console.log(comments))
  .catch((error) => console.error(error));
```

### 27. What is Promise.all?

```javascript
const promise1 = Promise.resolve(1);
const promise2 = Promise.resolve(2);
const promise3 = Promise.resolve(3);

Promise.all([promise1, promise2, promise3])
  .then((values) => console.log(values)) // [1, 2, 3]
  .catch((error) => console.error(error));
```

### 28. What is Promise.race?

```javascript
const promise1 = new Promise((resolve) => setTimeout(() => resolve(1), 1000));
const promise2 = new Promise((resolve) => setTimeout(() => resolve(2), 500));

Promise.race([promise1, promise2])
  .then((value) => console.log(value)) // 2
  .catch((error) => console.error(error));
```

## Object-Oriented Programming

### 29. What are classes?

```javascript
class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    console.log(`${this.name} makes a sound`);
  }
}

class Dog extends Animal {
  speak() {
    console.log(`${this.name} barks`);
  }
}

const dog = new Dog("Rex");
dog.speak(); // "Rex barks"
```

### 30. What are static methods?

```javascript
class MathUtils {
  static sum(a, b) {
    return a + b;
  }
}

console.log(MathUtils.sum(1, 2)); // 3
```

## Modules

### 31. What are ES6 modules?

```javascript
// math.js
export const sum = (a, b) => a + b;
export const multiply = (a, b) => a * b;

// app.js
import { sum, multiply } from "./math.js";
console.log(sum(1, 2)); // 3
```

### 32. What is dynamic import?

```javascript
// Dynamic import
button.addEventListener("click", async () => {
  const module = await import("./module.js");
  module.doSomething();
});
```

## Web APIs

### 33. What is the Fetch API?

```javascript
fetch("https://api.example.com/data")
  .then((response) => response.json())
  .then((data) => console.log(data))
  .catch((error) => console.error(error));
```

### 34. What is localStorage?

```javascript
// Save data
localStorage.setItem("name", "John");

// Get data
const name = localStorage.getItem("name");
console.log(name); // "John"

// Remove data
localStorage.removeItem("name");
```

## Performance

### 35. What is debouncing?

```javascript
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

const debouncedSearch = debounce(search, 300);
input.addEventListener("input", debouncedSearch);
```

### 36. What is throttling?

```javascript
function throttle(func, limit) {
  let inThrottle;
  return function executedFunction(...args) {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

const throttledScroll = throttle(handleScroll, 100);
window.addEventListener("scroll", throttledScroll);
```

## Security

### 37. What is XSS protection?

```javascript
function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const userInput = "<script>alert('xss')</script>";
const safeOutput = escapeHtml(userInput);
console.log(safeOutput); // "&lt;script&gt;alert('xss')&lt;/script&gt;"
```

### 38. What is CSRF protection?

```javascript
// Server-side
app.use((req, res, next) => {
  res.cookie("csrf-token", generateToken(), { httpOnly: true });
  next();
});

// Client-side
fetch("/api/data", {
  method: "POST",
  headers: {
    "X-CSRF-Token": getCookie("csrf-token"),
  },
});
```

## Testing

### 39. What is unit testing?

```javascript
// math.js
export function sum(a, b) {
  return a + b;
}

// math.test.js
test("adds 1 + 2 to equal 3", () => {
  expect(sum(1, 2)).toBe(3);
});
```

### 40. What is mocking?

```javascript
// api.js
export async function fetchData() {
  const response = await fetch("https://api.example.com/data");
  return response.json();
}

// api.test.js
jest.mock("node-fetch");
test("fetchData returns data", async () => {
  fetch.mockResolvedValue({ json: () => ({ data: "test" }) });
  const result = await fetchData();
  expect(result).toEqual({ data: "test" });
});
```

## Advanced Concepts

### 41. What is the event delegation?

```javascript
document.getElementById("parent").addEventListener("click", (e) => {
  if (e.target.matches("button")) {
    console.log("Button clicked:", e.target.textContent);
  }
});
```

### 42. What is the observer pattern?

```javascript
class Observable {
  constructor() {
    this.observers = [];
  }

  subscribe(observer) {
    this.observers.push(observer);
  }

  notify(data) {
    this.observers.forEach((observer) => observer(data));
  }
}

const observable = new Observable();
observable.subscribe((data) => console.log("Observer 1:", data));
observable.subscribe((data) => console.log("Observer 2:", data));
observable.notify("Hello!");
```

### 43. What is the singleton pattern?

```javascript
class Singleton {
  constructor() {
    if (!Singleton.instance) {
      Singleton.instance = this;
    }
    return Singleton.instance;
  }
}

const instance1 = new Singleton();
const instance2 = new Singleton();
console.log(instance1 === instance2); // true
```

### 44. What is the factory pattern?

```javascript
class Car {
  constructor(options) {
    this.wheels = options.wheels || 4;
    this.color = options.color || "black";
  }
}

class CarFactory {
  create(type) {
    switch (type) {
      case "sedan":
        return new Car({ wheels: 4 });
      case "suv":
        return new Car({ wheels: 6 });
      default:
        return new Car();
    }
  }
}

const factory = new CarFactory();
const sedan = factory.create("sedan");
```

### 45. What is the module pattern?

```javascript
const counter = (function () {
  let count = 0;

  return {
    increment() {
      count++;
    },
    getCount() {
      return count;
    },
  };
})();

counter.increment();
console.log(counter.getCount()); // 1
```

## Modern JavaScript Features

### 46. What are optional chaining and nullish coalescing?

```javascript
// Optional chaining
const user = {
  address: {
    street: "123 Main St",
  },
};
console.log(user?.address?.street); // "123 Main St"
console.log(user?.address?.zipCode); // undefined

// Nullish coalescing
const name = null ?? "Default";
console.log(name); // "Default"
```

### 47. What are private class fields?

```javascript
class Counter {
  #count = 0;

  increment() {
    this.#count++;
  }

  getCount() {
    return this.#count;
  }
}

const counter = new Counter();
counter.increment();
console.log(counter.getCount()); // 1
// console.log(counter.#count); // Error: Private field
```

### 48. What are top-level await?

```javascript
// In modules
const data = await fetch("https://api.example.com/data");
console.log(data);
```

### 49. What are dynamic imports?

```javascript
// Dynamic import
const module = await import("./module.js");
module.doSomething();
```

### 50. What are web workers?

```javascript
// main.js
const worker = new Worker("worker.js");
worker.postMessage("Hello");
worker.onmessage = (e) => {
  console.log(e.data);
};

// worker.js
self.onmessage = (e) => {
  const result = e.data.toUpperCase();
  self.postMessage(result);
};
```
