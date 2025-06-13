
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
from utils.losses import image_classification_loss # Used by CTM, LSTM


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument('--model', type=str, default='ctm', choices=['ctm', 'lstm', 'ff'], help='Model type to train.')

    # Model Architecture
    # Common
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='none', help='Type of backbone featureiser.')
    # parser.add_argument('--backbone_type', type=str, default='resnet18-4', help='Type of backbone featureiser.')
    # CTM / LSTM specific
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input (CTM, LSTM).')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads (CTM, LSTM).')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks (CTM, LSTM).')
    parser.add_argument('--positional_embedding_type', type=str, default='none', help='Type of positional embedding (CTM, LSTM).',
                        choices=['none',
                                 'learnable-fourier',
                                 'multi-learnable-fourier',
                                 'custom-rotational'])
    # CTM specific
    parser.add_argument('--synapse_depth', type=int, default=4, help='Depth of U-NET model for synapse. 1=linear, no unet (CTM only).')
    parser.add_argument('--n_synch_out', type=int, default=512, help='Number of neurons to use for output synch (CTM only).')
    parser.add_argument('--n_synch_action', type=int, default=512, help='Number of neurons to use for observation/action synch (CTM only).')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset (CTM only).')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch (CTM only).')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS (CTM only).')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory (CTM only).')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory (CTM only).')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs (CTM only).')
    # LSTM specific
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM stacked layers (LSTM only).')

    # Training
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=250, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile model components (backbone, synapses if CTM).')
    parser.add_argument('--num_workers_train', type=int, default=1, help='Num workers training.')

    # Housekeeping
    parser.add_argument('--log_dir', type=str, default='logs/runtwo', help='Directory for logging.')
    # parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use.')
    parser.add_argument('--dataset', type=str, default='grid_dif', help='Dataset to use.')
    parser.add_argument('--data_root', type=str, default='data/', help='Where to save dataset.')
    parser.add_argument('--save_every', type=int, default=100, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?')
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True, help='Should use strict reload for model weights.') # Added back
    parser.add_argument('--track_every', type=int, default=50, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='How many minibatches to approx metrics. Set to -1 for full eval')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')

    parser.add_argument('--model_path', type=str, 
                        default=r'C:\Users\abhay\OneDrive\IIITD\ARC\continuous-thought-machines\logs\grid_num\checkpoint.pt',
                        help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--dataset_path', type=str, 
                        default= r"C:\Users\abhay\OneDrive\IIITD\ARC\continuous-thought-machines\data\grid_num_class\grid_digit_label_dataset.json",
                        help='Path to the grid_digit_label_dataset.json file.')

    args = parser.parse_args()
    return args

# Add the project root to the Python path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError: # Fallback for environments where __file__ is not defined
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    if project_root not in sys.path:
         sys.path.insert(0, project_root)


from models.ctm_grid_simple import ContinuousThoughtMachineGridDiff
# from data.grid_diff.load_dataset import load_and_split_dataset
from data.grid_num_class.load_dataset import load_and_split_dataset


def visualize_sample(grid1, grid2, true_label, predicted_label, sample_idx):
    """
    Visualizes a pair of grids, their true label, and the predicted label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Ensure grids are on CPU and are numpy arrays
    grid1_np = grid1.cpu().numpy()
    grid2_np = grid2.cpu().numpy()
    
    axes[0].imshow(grid1_np, cmap='viridis', vmin=0, vmax=10) # Assuming values 0-10 as in generator
    axes[0].set_title("Grid 1")
    axes[0].axis('off')
    
    axes[1].imshow(grid2_np, cmap='viridis', vmin=0, vmax=10)
    axes[1].set_title("Grid 2")
    axes[1].axis('off')
    
    # true_label_item = true_label.item() if torch.is_tensor(true_label) else true_label
    # predicted_label_item = predicted_label.item() if torch.is_tensor(predicted_label) else predicted_label

    fig.suptitle(f"Sample {sample_idx}\\nTrue Label: {true_label} | Predicted Label: {predicted_label}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    out_dims = 10 if args.dataset == 'grid_dif' else 10 # Adjust based on dataset
    prediction_reshaper = [-1]  # Problem specific
    
    # Instantiate the model
    # Ensure these hyperparameters match the ones used for training
    model = ContinuousThoughtMachineGridDiff(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        ).to(device)

    # Load the trained model weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        # Try common keys for state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model state_dict from checkpoint['model_state_dict']")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded model state_dict from checkpoint['state_dict']")
            else:
                # Assuming the checkpoint itself is the state_dict
                model.load_state_dict(checkpoint)
                print(f"Loaded model state_dict directly from checkpoint")
        else:
            # If checkpoint is not a dict, assume it's the state_dict itself
            model.load_state_dict(checkpoint)
            print(f"Loaded model state_dict directly from checkpoint object")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure the checkpoint path is correct and the checkpoint format is compatible.")
        print("Commonly, checkpoints are dictionaries with a key like 'model_state_dict' or 'state_dict'.")
        return

    model.to(device)
    model.eval()

    # Load the dataset
    try:
        _, test_dataset = load_and_split_dataset(args.dataset_path, test_size=0.2, random_state=42)
        if test_dataset is None or len(test_dataset) == 0:
            print(f"Failed to load or test dataset is empty from {args.dataset_path}")
            return
        print(f"Loaded test dataset with {len(test_dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Visualize a few samples
    num_to_visualize = min(10, len(test_dataset))
    print(f"Visualizing {num_to_visualize} samples...")

    for i in range(num_to_visualize):
        sample = test_dataset[i]
        
        print(sample)
        
        grid1 = sample[0][0]
        grid2 = sample[0][1]
        true_label = sample[1]
        # Prepare inputs for the model
        grid1_batch = grid1.unsqueeze(0).to(device) # Add batch dimension: (1, H, W)
        grid2_batch = grid2.unsqueeze(0).to(device) # Add batch dimension: (1, H, W)

        label = torch.tensor(true_label, dtype=torch.long).to(device)  # Assuming label is a single integer for the pair
        input = sample[0] .unsqueeze(0).to(device)
        target = label.unsqueeze(0).to(device)  # Assuming target is a single label for the pair
        with torch.no_grad():
            # Assuming the model's forward pass takes two grids separately: model(grid1, grid2)
            # If your model expects a single stacked/concatenated input, adjust here.
            # E.g., stacked_grids = torch.stack((grid1_batch, grid2_batch), dim=1) # (B, 2, H, W)
            # output = model(stacked_grids)
            try:
                # output_logits = model(input)
                predictions, certainties, synchronisation = model(input)
                loss, where_most_certain = image_classification_loss(predictions, certainties, target, use_most_certain=True)
                # accuracy = (predictions.argmax(1)[torch.arange(predictions.size(0), device=predictions.device),where_most_certain] == target).float().mean()
                # class_preds = torch.argmax(certainties, dim=1)  
                
                max_per_class, _ = certainties.max(dim=2)     # → shape [B, C], max certainty over ticks for each class
                final_class   = max_per_class.argmax(dim=1)   # → shape [B], most certain class for each sample 

                last_tick_preds = predictions[:, :, -1]        # → shape [B, C], only the final tick
                final_class     = last_tick_preds.argmax(dim=1)
            except Exception as e:
                print(f"Error during model forward pass for sample {i}: {e}")
                print("Please check if the model's forward signature matches the input provided (two separate grids).")
                print("Input grid1 shape: ", grid1_batch.shape)
                print("Input grid2 shape: ", grid2_batch.shape)
                continue

        # print(output_logits[1])
        # print(output_logits[2])
        # Get predicted class
        # Assuming output_logits is of shape (batch_size, num_classes)
        # pred_prob = torch.softmax(predictions, dim=1)
        # predicted_label = torch.argmax(pred_prob, dim=1)
        print(true_label, final_class)

        visualize_sample(grid1, grid2, true_label, final_class, i)

if __name__ == '__main__':
    args = parse_args()

    main(args)
