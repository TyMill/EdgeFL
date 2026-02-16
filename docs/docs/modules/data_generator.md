# Data Generator

`edgefl.data.generator` provides synthetic environmental data utilities used in examples and tests.

## Main functions

- `generate_environmental_data(...)` → single dataframe with environmental variables and `aqi` target.
- `generate_mixed_environmental_data(...)` → mixture of region profiles in one dataset.
- `generate_clients_data(...)` → list of `(X, y)` tuples ready for `ClientNode`.
- `load_region_profile(path)` / `load_region_profiles(paths)` → load YAML/JSON profile definitions.

## Profiles and schema

Profiles are described with `RegionProfile`, `VariableParams`, and `TargetParams`.
They allow controlling:

- variable mean/std/min/max,
- optional seasonal components,
- feature correlations,
- target coefficients and noise.

## Example: generate client-ready arrays

```python
from edgefl.data import generate_clients_data

clients_data = generate_clients_data(
    n_clients=4,
    n_samples=120,
    seed=42,
)

X0, y0 = clients_data[0]
print(X0.shape, y0.shape)
```

## Example: custom region profile file

```python
from edgefl.data import load_region_profile, generate_environmental_data

profile = load_region_profile("edgefl/data/profiles/urban_winter.yaml")
df = generate_environmental_data(n_samples=200, profile=profile, seed=1)
print(df.head())
```
