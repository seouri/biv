# Architecture Overview

The `biv` package provides a modular framework for detecting Biologically Implausible Values (BIVs) in biomedical tabular data, specifically longitudinal weight and height measurements. Designed for public health and epidemiology research, it enables flexible cleaning of biometric data by applying multiple detection methods while maintaining a clean separation between the core orchestration logic and individual detection algorithms.

## High-Level Architecture

The architecture follows a modular, plugin-like pattern where detection methods are organized as independent modules under a dedicated `methods` directory. Each method implements a common interface, enabling the core system to invoke them uniformly. This design ensures that new detection methods can be added without modifying the core logic.

```
biv/
├── biv/
│   ├── __init__.py      # Exposes public API (detect, remove)
│   ├── api.py           # Public functions and orchestrator logic
│   └── methods/
│       ├── __init__.py      # Detector registry and registration mechanism
│       ├── base.py          # Abstract base class for all detectors
│       └── [method_name]/
│           ├── __init__.py
│           └── detector.py    # Implements [MethodName]Detector class
├── tests/
│   ├── conftest.py
│   ├── test_api.py        # Tests for main API functions
│   └── methods/
│       └── test_[method_name]/
│           └── test_detector.py # Method-specific tests
├── [other project files...]
```

## Core Components

### API Layer (`api.py`)
The `api.py` module serves as the primary entry point and orchestrator:
- Exposes the public functions `detect()` and `remove()`
- Manages the orchestration of multiple detection methods
- Handles configuration parsing and detector instantiation
- Combines results from different methods using logical operations

### Methods Registry (`methods/__init__.py`)
Provides automatic registration of detection methods:
- Mapping from method names to detector classes
- Enables low-friction addition of new methods
- Allows the API layer to discover available methods dynamically

### Base Detector (`methods/base.py`)
Defines the interface contract for all detection methods:
- `BaseDetector` abstract class with standard methods
- Initialization with method-specific configuration parameters using Pydantic for type-safe config handling and clear validation errors
- Parameter validation interface with Pydantic models
- Unified detection signature returning boolean flags

### Detection Methods
Each detection method is implemented in its own module (e.g., `methods/range/`, `methods/zscore/`):
- Inherits from `BaseDetector`
- Implements detection logic specific to the method
- Handles its own parameter validation
- Returns DataFrame with boolean outlier flags

## Method Extensibility

Adding a new detection method requires minimal changes to the core system:

1. **Create the method module** under `biv/methods/[new_method]/`
   - Implement `[NewMethod]Detector` class inheriting from `BaseDetector`
   - Define initialization and validation logic
   - Implement the `detect()` method

2. **The method is auto-registered** via introspection in `methods/__init__.py`
   - Automatic discovery of detector classes inheriting from `BaseDetector`
   - No manual registry update required

That's it—the core system remains unchanged, and the new method is automatically available via the public API.

## Configuration Flow

Configuration flows from the user through the API layer to individual detectors:

1. User calls `biv.detect(dataframe, methods={'range': {'min': 20, 'max': 200}, 'zscore': {'threshold': 3}})`

2. `api.py` iterates through each method name and config pair

3. Uses registry to find appropriate detector class

4. Instantiates detector with method-specific config kwargs

5. Invokes `detect()` on each instance, collecting boolean flags

6. Combines flags according to the specified logical combination

## Testing Architecture

The test structure mirrors the source code organization:
- `test_api.py`: Tests end-to-end API behavior and orchestration
- `test_[method_name]/test_detector.py`: Method-specific unit tests
- Performance profiling tests for scalability and efficiency
- Ensures changes to individual methods don't break integration

## Design Principles

- **Modularity**: Physical separation by directory prevents code entanglement
- **Interface-Driven**: Clear contracts ensure consistency and predictability
- **Separation of Concerns**: Orchestration logic isolated from algorithm details
- **Registry Pattern**: Enables loose coupling and automatic discovery
- **Configuration Decoupling**: Methods receive only their relevant parameters

This architecture provides a robust, maintainable, and highly extensible foundation for Biologically Implausible Values (BIV) detection, supporting future growth and contributor onboarding.
