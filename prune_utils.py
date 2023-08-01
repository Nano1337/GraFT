import torch
import torch.nn as nn
from typing import Optional

# MLPs to prune is a list of tuples, each tuple contains two linear layers, and we prune the neurons in between them
def get_indices_to_prune(MLPs_to_prune, num_neurons_to_prune, criteria = 'Magnitude'):
    if criteria == 'Magnitude':
        #Get a list of lists of all the hidden neurons' magnitudes
        magnitudes = []
        for linear1, linear2 in MLPs_to_prune:
            magnitude1 = torch.sum(torch.abs(linear1.weight), dim = 1) # get magnitudes along the input dim
            magnitude2 = torch.sum(torch.abs(linear2.weight), dim = 0) # get magnitudes along the output dim
            hidden_magnitudes = magnitude1 + magnitude2 # get overall magnitude for the hidden neurons
            magnitudes.append(hidden_magnitudes)
        result = []

        # Start and end index for each tensor in the list
        start_index = 0
        end_index = 0
        
        # Concat all the magnitudes across blocks, then find the lowest k magnitudes to prune
        all_values = torch.cat(magnitudes)
        _, indices = torch.topk(all_values, num_neurons_to_prune, largest=False)
        
        # Reorder the indices back into the shape of "magnitudes" variable, the tensor should be empty for blocks with no indices to prune
        for hidden_magnitudes in magnitudes:
            end_index += hidden_magnitudes.numel()  # Update the end index
            tensor_indices = indices[(indices >= start_index) & (indices < end_index)]
            tensor_indices -= start_index
            result.append(tensor_indices)
            start_index = end_index
        
        return result
    else:
        pass

# Helper function to prune individual weight matrices    
def remove_indices(tensor, indices, axis):
    device = tensor.device  # Get the device of the tensor
    mask = torch.ones(tensor.shape[axis], dtype=torch.bool).to(device)  # Ensure the mask is on the same device
    mask[indices] = 0
    indices_to_keep = torch.arange(tensor.shape[axis]).to(device)[mask]  # Ensure indices are on the same device
    new_tensor = tensor.index_select(axis, indices_to_keep)
    return new_tensor

# Overall prune step for the entire model, prunes in place.
def prune_model(model, fabric, num_neurons_to_prune = 0, num_heads_to_prune = 0):
    if num_neurons_to_prune != 0:
        # Grab weights to measure importance
        MLPs_to_prune = []
        for layer in model.transformer.encoder.layer:
            MLPs_to_prune.append( (layer.intermediate.dense, layer.output.dense) )

        # indices: list of tensors, len(indices) == num blocks in model, each tensor represents the indices in that block to prune
        indices = get_indices_to_prune(MLPs_to_prune, num_neurons_to_prune, criteria = 'Magnitude')

        # Prune MLPs at those hidden indices
        for idx, hidden_indices in enumerate(indices):
            mlp_in = model.transformer.encoder.layer[idx].intermediate.dense
            mlp_out = model.transformer.encoder.layer[idx].output.dense

            #Initialize New Linear layers with less weights
            pruned_mlp_in = nn.Linear(mlp_in.weight.size()[1], mlp_in.weight.size()[0] - len(hidden_indices))
            pruned_mlp_out = nn.Linear(mlp_out.weight.size()[1] - len(hidden_indices), mlp_out.weight.size()[0])

            # Prune Biases
            pruned_mlp_in.bias = torch.nn.Parameter(remove_indices(mlp_in.bias, hidden_indices, axis=0))
            pruned_mlp_out.bias = torch.nn.Parameter(mlp_out.bias)

            # Prune Weights
            pruned_mlp_in.weight = torch.nn.Parameter(remove_indices(mlp_in.weight, hidden_indices, axis=0))
            pruned_mlp_out.weight = torch.nn.Parameter(remove_indices(mlp_out.weight, hidden_indices, axis=1))
            
            model.transformer.encoder.layer[idx].intermediate.dense = pruned_mlp_in.to(fabric.device)
            model.transformer.encoder.layer[idx].output.dense = pruned_mlp_out.to(fabric.device)

    if num_heads_to_prune != 0:
        pass
