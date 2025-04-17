# Data Generator

The `generator.py` module provides synthetic environmental data for simulating IoT sensors in different city regions.

### Functions
- `generate_environmental_data(n_samples, region_factor, seed)`: Returns a DataFrame with temperature, humidity, PM2.5, and AQI.
- `generate_clients_data(n_clients, n_samples)`: Returns a list of (X, y) datasets per client with varied region characteristics.
