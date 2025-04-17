import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

def generate_synthetic_data(n_clients=5, n_samples=100):
    clients = []
    for client_id in range(n_clients):
        X = np.random.rand(n_samples, 3) * (1 + 0.1 * client_id)
        noise = np.random.normal(0, 0.1, n_samples)
        y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + noise
        clients.append(ClientNode(client_id, X, y))
    return clients


def run_simulation(rounds=5):
    clients = generate_synthetic_data()
    server = FederatedServer(clients)

    for r in range(rounds):
        print(f"--- Round {r + 1} ---")
        server.distribute_and_train()

    # Generate test set
    X_test = np.random.rand(100, 3)
    y_test = 3 * X_test[:, 0] + 2 * X_test[:, 1] + X_test[:, 2] + np.random.normal(0, 0.1, 100)
    mse = server.evaluate_global_model(X_test, y_test)
    print(f"Final MSE on test set: {mse:.4f}")


if __name__ == "__main__":
    run_simulation()
