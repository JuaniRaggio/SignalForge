# mkt-PreD
AI-based market forecasting tool

## Architecture and technologies
---
### Concurrency -> Go
---
Goroutines makes this election almost indisputable.
1. Easy to develop concurrent tasks
2. Escalability
3. Eficiency -> In terms of execution time and memory resources

### AI for predictions -> Python
---
Has a widely used variety of libs which makes the developing experience easier. After having a functional product we could migrate this part
of the software to C++ if needed since we will have to do lots of operations and we might have to optimize it.

### Web scraping -> TBD
---
Options:
- Python
    #### Pros:
        + Easy to develop
        + Wide variety of libs, no proxy needed for HTTP requests
    #### Cons:
        - GIL limits the concurrent scraping
        - Its not possible to execute multiple python programs concurrently -> Not eficient
- C++
    #### Pros:
        + Lightweight
        + Max optimization
    #### Neutral:
        * The amount of libraries is intermidiate, not as wide as python's
    #### Cons:
        - Difficulty
        - Manual management of proxies for rotational requests
- Go
    #### Pros:
        + Full Go stack
        + Easy to debug and maintain
    #### Cons:
        - Limited scraping tools
    #### Neutral:
        * Crafting a scraping library might be a good idea for the Go community


