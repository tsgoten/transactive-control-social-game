import torch
import torch.nn as nn
class PFL_Hypernet(nn.Module):
    def __init__(self, n_nodes, embedding_dim, num_layers, num_hidden, out_params_path, lr):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.lr = lr
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim

        f = open(out_params_path, "r")
        param_string = f.read()
        self.param_shapes = eval(param_string)
        self.out_dim = 0

        validate_inputs(n_nodes, embedding_dim, num_layers, num_hidden, lr)
        
        self.embedding = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        self.layers = [self.embedding]
        if num_layers == 1:
            self.layers.append(nn.Linear(embedding_dim, self.out_dim))
        else:
            self.layers.append(nn.Linear(embedding_dim, num_hidden))
            for i in range(1, num_layers - 1):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(nn.ReLu())
            self.layers.append(nn.Linear(num_hidden, self.out_dim))

        self.net = nn.Sequential(*self.layers)
        

        
    def validate_inputs(n_nodes, embedding_dim, num_layers, num_hidden, out_dim, lr):
        assert n_nodes > 0, "n_nodes <= 0"
        assert isinstance(n_nodes, int) == True, "n_nodes must be an int"
        assert embedding_dim > 0, "embedding_dim <= 0"
        assert isinstance(embedding_dim, int) == True, "embedding_dim must be an int"
        assert num_layers > 0, "num_layers <= 0"
        assert isinstance(num_layers, int) == True, "num_layers must be an int"
        assert num_hidden > 0, "num_hidden <= 0"
        assert isinstance(num_hidden, int) == True, "num_hidden must be an int"
        assert lr > 0, "lr <= 0"
        
    def forward(self, x):
        return self.net