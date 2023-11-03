## torch_erf

A torch module implementing the [complex error function](https://en.wikipedia.org/wiki/Error_function), supporting GPU computation, differentaible, and compatible
with any tensor shape.

### Installation

```bash
pip install torch_erf@git+https://github.com/redsnic/torch_erf/
```

### Usage

```python
from torch_erf.ERF import ERF_1994 
```

see man/ERF_testing.ipynb

> the package depends on `torch`. However, `numpy`, `scipy` and `matplotlib` are required for the tests. 

