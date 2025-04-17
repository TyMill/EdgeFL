import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from edgefl.data.generator import generate_clients_data


class ClientNode:
    def __init__(self, client_id, X, y):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.model = LinearRegression()

    def train_local_model(self):
        self.model.fit(self.X, self.y)
        return self.get_model_weights()

    def get_model_weights(self):
        return np.append(self.model.coef_, self.model.intercept_)

    def set_model_weights(self, weights):
        self.model.coef_ = weights[:-1]
        self.model.intercept_ = weights[-1]


class FederatedServer:
    def __init__(self, clients):
        self.clients = clients
        self.global_weights = None

    def aggregate_weights(self, all_weights):
        return np.mean(all_weights, axis=0)

    def distribute_and_train(self):
        all_weights = []
        for client in self.clients:
            if self.global_weights is not None:
                client.set_model_weights(self.global_weights)
            weights = client.train_local_model()
            all_weights.append(weights)
        self.global_weights = self.aggregate_weights(all_weights)

    def evaluate_global_model(self, X_test, y_test):
        dummy_model = LinearRegression()
        dummy_model.coef_ = self.global_weights[:-1]
        dummy_model.intercept_ = self.global_weights[-1]
        predictions = dummy_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse


# === Example Simulation ===

def run_simulation(rounds=5):
    clients_data = generate_clients_data(n_clients=5, n_samples=100)
    clients = [ClientNode(i, X, y) for i, (X, y) in enumerate(clients_data)]
    server = FederatedServer(clients)

    for r in range(rounds):
        print(f"--- Round {r + 1} ---")
        server.distribute_and_train()

    # Generate test set from average region factor (1.2)
    from edgefl.data.generator import generate_environmental_data
    df_test = generate_environmental_data(n_samples=100, region_factor=1.2, seed=999)
    X_test = df_test[['temperature', 'humidity', 'pm25']].values
    y_test = df_test['aqi'].values
    mse = server.evaluate_global_model(X_test, y_test)
    print(f"Final MSE on test set: {mse:.4f}")


if __name__ == "__main__":
    run_simulation()
