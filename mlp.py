import numpy as np

class MLP:
    def __init__(self, layers, learning_rate=0.01):
        """
        Inicializa a MLP
        
        Args:
            layers: lista com o número de neurônios em cada camada
                   ex: [2, 4, 3, 1] = entrada(2), oculta1(4), oculta2(3), saída(1)
            learning_rate: taxa de aprendizado
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # Inicializar pesos e bias aleatoriamente
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Função de ativação sigmoid"""
        # Clipping para evitar overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivada da função sigmoid"""
        return x * (1 - x)
    
    def relu(self, x):
        """Função de ativação ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivada da função ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """
        Forward pass através da rede
        
        Args:
            X: dados de entrada (batch_size, input_features)
            
        Returns:
            output: saída da rede
            activations: lista com as ativações de cada camada
        """
        activations = [X]
        current_input = X
        
        for i in range(self.num_layers - 1):
            # Calcular z = X * W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            # Aplicar função de ativação
            if i < self.num_layers - 2:  # Camadas ocultas usam ReLU
                activation = self.relu(z)
            else:  # Camada de saída usa sigmoid
                activation = self.sigmoid(z)
            
            activations.append(activation)
            current_input = activation
        
        return current_input, activations
    
    def backward(self, X, y, activations):
        """
        Backward pass (backpropagation)
        
        Args:
            X: dados de entrada
            y: rótulos verdadeiros
            activations: ativações de cada camada do forward pass
        """
        m = X.shape[0]  # número de exemplos
        
        # Calcular erro da camada de saída
        output_error = activations[-1] - y
        
        # Listas para armazenar gradientes
        weight_gradients = []
        bias_gradients = []
        
        # Backpropagation
        current_error = output_error
        
        for i in range(self.num_layers - 2, -1, -1):
            # Gradientes dos pesos e bias
            dW = np.dot(activations[i].T, current_error) / m
            db = np.mean(current_error, axis=0, keepdims=True)
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            if i > 0:  # Não calcular erro para a camada de entrada
                # Calcular erro da camada anterior
                if i == self.num_layers - 2:  # Vindo da camada de saída (sigmoid)
                    current_error = np.dot(current_error, self.weights[i].T) * self.relu_derivative(activations[i])
                else:  # Vindo de camada oculta (ReLU)
                    current_error = np.dot(current_error, self.weights[i].T) * self.relu_derivative(activations[i])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Atualizar pesos e bias usando gradiente descendente"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def compute_loss(self, y_true, y_pred):
        """Calcular loss usando binary cross-entropy"""
        # Evitar log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Treinar a rede neural
        
        Args:
            X: dados de entrada
            y: rótulos
            epochs: número de épocas
            verbose: imprimir progresso
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output, activations = self.forward(X)
            
            # Calcular loss
            loss = self.compute_loss(y, output)
            losses.append(loss)
            
            # Backward pass
            weight_gradients, bias_gradients = self.backward(X, y, activations)
            
            # Atualizar parâmetros
            self.update_parameters(weight_gradients, bias_gradients)
            
            # Imprimir progresso
            if verbose and epoch % 100 == 0:
                print(f'Época {epoch}, Loss: {loss:.4f}')
        
        return losses
    
    def predict(self, X):
        """Fazer predições"""
        output, _ = self.forward(X)
        return (output > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Retornar probabilidades"""
        output, _ = self.forward(X)
        return output