# FederatedServer

The `FederatedServer` coordinates federated learning across multiple `ClientNode` instances.

### Features
- Distributes the latest global model weights to all clients
- Aggregates updated weights using Federated Averaging
- Evaluates global model on a test set using MSE

### Methods
- `distribute_and_train()`: Pushes global weights, collects client updates.
- `aggregate_weights(all_weights)`: Averages weights across all clients.
- `evaluate_global_model(X_test, y_test)`: Calculates test MSE.
