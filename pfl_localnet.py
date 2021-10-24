import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()

# The global, shared hypernet to be used by all models.
SHARED_HNET = None # TODO: Set this according to command line arguments


class TorchSharedWeightsModel(TorchModelV2, nn.Module):
    """Example of weight sharing between two different TorchModelV2s.
    The shared (single) layer is simply defined outside of the two Models,
    then used by both Models in their forward pass.
    """

    def __init__(self, observation_space, action_space, num_outputs,
                 model_config, name):
        TorchModelV2.__init__(self, observation_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Non-shared initial layer.
        self.first_layer = SlimFC(
            int(np.product(observation_space.shape)),
            64,
            activation_fn=nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_)

        # Non-shared final layer.
        self.last_layer = SlimFC(
            64,
            self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_)
        self.vf = SlimFC(
            64,
            1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self._global_shared_layer = TORCH_GLOBAL_SHARED_LAYER
        self._output = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        out = self.first_layer(input_dict["obs"])
        self._output = self._global_shared_layer(out)
        model_out = self.last_layer(self._output)
        return model_out, []

    @override(ModelV2)
    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.vf(self._output), [-1])