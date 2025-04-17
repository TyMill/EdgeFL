import numpy as np
import pandas as pd

def generate_environmental_data(n_samples=100, region_factor=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Czujniki środowiskowe: temperatura, wilgotność, jakość powietrza (PM2.5)
    temperature = 15 + 10 * np.random.rand(n_samples) * region_factor
    humidity = 30 + 20 * np.random.rand(n_samples) * region_factor
    pm25 = 5 + 25 * np.random.rand(n_samples) * (2 - region_factor)

    # Funkcja celu (regresja): indeks jakości powietrza (AQI)
    # Zakładamy, że wyższa temperatura i PM2.5 = gorszy AQI, wilgotność łagodzi
    noise = np.random.normal(0, 5, n_samples)
    aqi = 0.5 * temperature + 0.3 * pm25 - 0.2 * humidity + noise

    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pm25': pm25,
        'aqi': aqi
    })
    return df

def generate_clients_data(n_clients=5, n_samples=100):
    client_datasets = []
    for i in range(n_clients):
        region_factor = 1.0 + (i * 0.1)  # różne warunki dla różnych lokalizacji
        df = generate_environmental_data(n_samples=n_samples, region_factor=region_factor, seed=42 + i)
        X = df[['temperature', 'humidity', 'pm25']].values
        y = df['aqi'].values
        client_datasets.append((X, y))
    return client_datasets
