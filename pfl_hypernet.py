import torch
import torch.nn as nn
from collections import OrderedDict

class PFL_Hypernet(nn.Module):
    def __init__(self, n_nodes, embedding_dim, num_layers, num_hidden, out_params_path, lr, device):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.lr = lr
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.device = device
        

        f = open(out_params_path, "r")
        param_string = f.read()
        self.out_params_dict = eval(param_string)
        self.out_dim = self.calculate_out_dim()
        shifts = nn.Parameter(torch.zeros(self.out_dim, device=self.device))
        scales = nn.Parameter(torch.ones(self.out_dim, device=self.device))
        self.register_parameter(name='shifts', param=shifts)
        self.register_parameter(name='scales', param=scales)
        self.validate_inputs(n_nodes, embedding_dim, num_layers, num_hidden, lr)
        
        self.embedding = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        self.layers = [self.embedding]
        if num_layers == 1:
            self.layers.append(nn.Linear(embedding_dim, self.out_dim))
        else:
            self.layers.append(nn.Linear(embedding_dim, num_hidden))
            for i in range(1, num_layers - 1):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(num_hidden, self.out_dim))
        self.net = nn.Sequential(*self.layers).to(device)
        
    def calculate_out_dim(self):
        dim = 0
        self.products_dict = {}
        for k, v in self.out_params_dict.items():
            product = 1
            for i in range(len(v)):
                product *= v[i]
            dim += product
            self.products_dict[k] = product
        return dim

    def create_weight_dict(self, weight_vector):
        weight_vector = weight_vector.squeeze()
        return_dict = OrderedDict()
        index = 0
        for k in self.out_params_dict.keys():

            # restrict weights to have mean 0 and variance scaled with avg of fan_in and fan_out, similar to Xavier initialization
           
           weights = weight_vector[index:index + self.products_dict[k]].reshape(self.out_params_dict[k])
           shifts = self.shifts[index:index + self.products_dict[k]].reshape(self.out_params_dict[k])
           scales = self.scales[index:index + self.products_dict[k]].reshape(self.out_params_dict[k])
           if len(self.out_params_dict[k]) > 1:
            fan_in = self.out_params_dict[k][0]
            fan_out = self.out_params_dict[k][1]
            mean_w = weights.mean()
            std_w = weights.std()
            weights = (weights - mean_w) / std_w
            
            weights *= scales * ((2 / (fan_in + fan_out)) ** 0.5)
            weights += shifts

           return_dict[k] = weights.to("cpu")
           
           index += self.products_dict[k]
        return return_dict
        
    def validate_inputs(self, n_nodes, embedding_dim, num_layers, num_hidden, lr):
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
        return self.create_weight_dict(self.net(torch.tensor(x).to(self.device)))