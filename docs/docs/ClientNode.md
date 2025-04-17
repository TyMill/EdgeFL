# ClientNode

The `ClientNode` class simulates an edge device in a federated learning environment. Each client:

- Stores its own local dataset `(X, y)`
- Trains a local model (Linear Regression)
- Returns weights to the federated server
- Can receive global weights and continue training

### Methods
- `train_local_model()`: Fits the model on local data and returns weights.
- `get_model_weights()`: Returns a flattened array of model coefficients + intercept.
- `set_model_weights(weights)`: Updates local model with global weights.
