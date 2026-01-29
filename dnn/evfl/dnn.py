import numpy as np

class DNN:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, learning_rate, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.activation = activation
        
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for all layers randomly
        layer_sizes = [self.input_size] + [self.neurons_per_layer] * self.hidden_layers + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i+1], layer_sizes[i])
            bias_vector = np.random.randn(layer_sizes[i+1], 1)
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def feedforward(self, x):
        activations = [x]
        
        for i in range(self.hidden_layers + 1):
            weighted_sum = np.dot(self.weights[i], activations[i]) + self.biases[i]
            activation_output = self.activation(weighted_sum)
            activations.append(activation_output)
        
        return activations[-1]
    
    def backpropagation(self, x, y):
        gradients_w = [np.zeros(weight.shape) for weight in self.weights]
        gradients_b = [np.zeros(bias.shape) for bias in self.biases]
        
        # Feedforward
        activations = [x]
        weighted_sums = []
        
        for i in range(self.hidden_layers + 1):
            weighted_sum = np.dot(self.weights[i], activations[i]) + self.biases[i]
            weighted_sums.append(weighted_sum)
            activation_output = self.activation(weighted_sum)
            activations.append(activation_output)
        
        # Backpropagation
        delta = (activations[-1] - y) * self.activation_derivative(weighted_sums[-1])
        gradients_w[-1] = np.dot(delta, activations[-2].T)
        gradients_b[-1] = delta
        
        for i in range(self.hidden_layers - 1, -1, -1):
            delta = np.dot(self.weights[i+1].T, delta) * self.activation_derivative(weighted_sums[i])
            gradients_w[i] = np.dot(delta, activations[i].T)
            gradients_b[i] = delta
        
        return gradients_w, gradients_b
    
    def load_data(partition_id: int, num_partitions: int, batch_size: int):
        """Load partition CIFAR10 data."""
        # Only initialize `FederatedDataset` once
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        # Construct dataloaders
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader
    
    def load_centralized_dataset():
        """Load test set and return dataloader."""
        # Load entire test set
        test_dataset = load_dataset("uoft-cs/cifar10", split="test")
        dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
        return DataLoader(dataset, batch_size=128)
    
    def train(self, x_train, y_train, max_iterations=100, convergence_threshold=1e-6):
        iterations = 0
        prev_loss = float('inf')
        
        while iterations < max_iterations:
            total_loss = 0
            
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]
                
                # Feedforward and backpropagation
                gradients_w, gradients_b = self.backpropagation(x, y)
                
                # Update weights and biases using gradient descent
                self.weights = [weight - self.learning_rate * grad_w for weight, grad_w in zip(self.weights, gradients_w)]
                self.biases = [bias - self.learning_rate * grad_b for bias, grad_b in zip(self.biases, gradients_b)]
                
                # Compute loss
                prediction = self.feedforward(x)
                loss = 0.5 * np.sum((prediction - y) ** 2)
                total_loss += loss
            
            # Check for convergence
            if abs(prev_loss - total_loss) < convergence_threshold:
                break
            
            # ```python
            iterations += 1
            prev_loss = total_loss
        
        print(f"Training completed in {iterations} iterations")
        return prev_loss

    def test(net, testloader, device):
        """Validate the model on the test set."""
        net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy
    
    def predict(self, x):
        return self.feedforward(x)