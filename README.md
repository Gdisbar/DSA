# System Design

[Process Flow](https://igotanoffer.com/blogs/tech/system-design-interview-questions)

[LLD Qns](https://github.com/ashishps1/awesome-low-level-design)

[Youtube](https://www.youtube.com/playlist?list=PLQEaRBV9gAFvzp6XhcNFpk1WdOcyVo9qT)


Below is a **ranked list of the 23 Gang of Four (GoF) design patterns** based on:

* **How commonly they appear in real production systems today**
* **How often backend engineers (FastAPI / microservices / distributed systems) encounter them**
* **Practical situations where they're used**

---

# ✅ **TOP TIER (Used ALL the time in industry)**

### ⭐ These 7 patterns are everywhere — APIs, microservices, DB layers, logging, auth, messaging, etc.

---

## **1. Singleton (Very Common)**

**Use when:** You need exactly *one instance* of something globally.

**Examples:**

* DB connection pool
* Redis client
* Logger instance
* Configuration loader

---

## **2. Strategy (Extremely Common)**

**Use when:** You want to switch algorithms/behaviors at runtime.

**Examples:**

* Payment processors (Stripe / PayPal / Razorpay)
* Authentication methods (JWT / OAuth / API key)
* Different caching strategies
* Sorting/comparison logic

---

## **3. Factory / Factory Method**

**Use when:** Object creation depends on conditions, environment, config.

**Examples:**

* Creating DB clients (Postgres vs MongoDB)
* Creating queue producers (Kafka vs RabbitMQ)
* Choosing cache backend (Redis vs in-memory)

---

## **4. Builder (Very Common in API/SDK development)**

**Use when:** A complex object requires step-by-step construction.

**Examples:**

* Constructing HTTP requests
* Building SQL queries (query builders)
* Creating complex Pydantic models
* Building Kafka producer configs

---

## **5. Observer / Pub-Sub (Very Common in Distributed Systems)**

**Use when:** One event should notify many subscribers.

**Examples:**

* Kafka consumers
* Redis pub-sub
* Webhooks
* Event-driven microservices

---

## **6. Adapter (Common with external APIs/services)**

**Use when:** You need to convert one interface into another.

**Examples:**

* Integrating 3rd-party APIs (Stripe, AWS, Twilio)
* Converting legacy code interfaces
* Making incompatible libraries work together

---

## **7. Decorator (Very Common in Python especially!)**

**Use when:** You want to add behavior to functions without modifying them.

**Examples:**

* FastAPI routes (`@app.get`)
* Logging decorators
* Caching (like @lru_cache)
* Authorization wrappers

---

# ⚡ **MID TIER (Common but less frequent — used in specific places)**

### ⭐ Useful in API design, databases, large-scale projects.

---

## **8. Proxy**

**Use when:** You want a wrapper that controls access to another object.

**Examples:**

* API gateway / reverse proxies
* Lazy loading DB objects
* Access control for resources

---

## **9. Facade**

**Use when:** You need to simplify a complex system behind a clean interface.

**Examples:**

* SDK clients
* Wrapping AWS services under a simple API
* Providing a single API over multiple microservices

---

## **10. Command**

**Use when:** You want to encapsulate actions or operations as objects.

**Examples:**

* Task/job scheduling (Celery tasks, command handlers)
* Undo/redo systems
* Event sourcing commands

---

## **11. Template Method**

**Use when:** You have a base algorithm but want to customize steps.

**Examples:**

* Authentication pipelines
* ETL pipelines
* Data processing workflows

---

## **12. Mediator**

**Use when:** Multiple components need to communicate without knowing each other directly.

**Examples:**

* Message brokers for microservices
* WebSocket manager
* GUI systems (less common in backend)

---

## **13. Chain of Responsibility**

**Use when:** Requests pass through a chain of handlers until one handles it.

**Examples:**

* Middleware pipelines (FastAPI, Express, Spring)
* Authentication → authorization → validation
* Logging chains

---

## **14. Composite**

**Use when:** You have hierarchical structures (tree-like).

**Examples:**

* Filesystem-like data
* Menu structures
* Organization hierarchy

---

## **15. State**

**Use when:** An object changes behavior depending on internal state.

**Examples:**

* Order/status transitions (Pending → Paid → Shipped)
* Authentication flows
* Streaming pipelines

---

# 🔽 **LOWER TIER (Used rarely — useful in niche cases)**

### ⭐ Typically used in games, GUIs, compilers, or specialized domains.

---

## **16. Iterator**

Already built into nearly every language — rarely implemented manually.

**Examples:**

* Custom iteration over DB results
* Streaming large files

---

## **17. Flyweight**

**Use when:** You need to share memory-heavy objects efficiently.

**Examples:**

* Caching huge objects
* Large-scale text rendering
* Game development

---

## **18. Memento**

**Use when:** You need to snapshot and restore object state.

**Examples:**

* Undo functionality
* Save game state

---

## **19. Interpreter**

**Use when:** Building simple scripting languages or expression evaluators.

**Examples:**

* SQL interpreters
* mini-languages like feature flags grammar

---

## **20. Prototype**

**Use when:** Object creation is expensive and you want cloning.

**Examples:**

* Cloning large config objects
* Pre-configured template objects

---

## **21. Bridge**

**Use when:** You want to separate abstraction from implementation.

**Examples:**

* Multi-DB drivers
* UI toolkits
* Device drivers

---

## **22. Visitor**

**Use when:** You want to operate on objects in a composite structure with many operations.

**Examples:**

* Compilers/AST traversal
* Analytics on hierarchical data

---

## **23. Abstract Factory (Rare unless you work with complex UI libraries)**

**Use when:** You want factories for related families of objects.

**Examples:**

* UI theme generators
* Complex plugin systems
* Modular architecture frameworks

---

# 🎯 **FINAL RANKING (Most → Least Used in Industry)**

### **TOP TIER (Use immediately)**

1. Singleton
2. Strategy
3. Factory / Factory Method
4. Builder
5. Observer / Pub-Sub
6. Adapter
7. Decorator

### **MID TIER (Know conceptually)**

8. Proxy
9. Facade
10. Command
11. Template Method
12. Mediator
13. Chain of Responsibility
14. Composite
15. State

### **LOWER TIER (Learn only if needed)**

16. Iterator
17. Flyweight
18. Memento
19. Interpreter
20. Prototype
21. Bridge
22. Visitor
23. Abstract Factory

---



