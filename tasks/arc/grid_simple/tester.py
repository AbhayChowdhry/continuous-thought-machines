
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt

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
from data.grid_diff.load_dataset import load_and_split_dataset

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
    
    true_label_item = true_label.item() if torch.is_tensor(true_label) else true_label
    predicted_label_item = predicted_label.item() if torch.is_tensor(predicted_label) else predicted_label

    fig.suptitle(f"Sample {sample_idx}\\nTrue Label: {true_label_item} | Predicted Label: {predicted_label_item}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    # Ensure these hyperparameters match the ones used for training
    model = ContinuousThoughtMachineGridDiff(
        d_input=args.d_input,
        d_model=args.d_model,
        heads=args.heads,
        iterations=args.iterations,
        num_classes=args.num_classes, # For grid_diff, this should be 2 (same/different)
        dropout=args.dropout,
        backbone_type=args.backbone_type
    )

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
        _, test_dataset = load_and_split_dataset(args.dataset_path, test_size=args.test_split_ratio, random_state=42)
        if test_dataset is None or len(test_dataset) == 0:
            print(f"Failed to load or test dataset is empty from {args.dataset_path}")
            return
        print(f"Loaded test dataset with {len(test_dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Visualize a few samples
    num_to_visualize = min(args.num_samples_visualize, len(test_dataset))
    print(f"Visualizing {num_to_visualize} samples...")

    for i in range(num_to_visualize):
        sample = test_dataset[i]
        grid1 = sample['grid1']
        grid2 = sample['grid2']
        true_label = sample['label']

        # Prepare inputs for the model
        grid1_batch = grid1.unsqueeze(0).to(device) # Add batch dimension: (1, H, W)
        grid2_batch = grid2.unsqueeze(0).to(device) # Add batch dimension: (1, H, W)

        with torch.no_grad():
            # Assuming the model's forward pass takes two grids separately: model(grid1, grid2)
            # If your model expects a single stacked/concatenated input, adjust here.
            # E.g., stacked_grids = torch.stack((grid1_batch, grid2_batch), dim=1) # (B, 2, H, W)
            # output = model(stacked_grids)
            try:
                output_logits = model(grid1_batch, grid2_batch)
            except Exception as e:
                print(f"Error during model forward pass for sample {i}: {e}")
                print("Please check if the model's forward signature matches the input provided (two separate grids).")
                print("Input grid1 shape: ", grid1_batch.shape)
                print("Input grid2 shape: ", grid2_batch.shape)
                continue


        # Get predicted class
        # Assuming output_logits is of shape (batch_size, num_classes)
        pred_prob = torch.softmax(output_logits, dim=1)
        predicted_label = torch.argmax(pred_prob, dim=1)

        visualize_sample(grid1, grid2, true_label, predicted_label.squeeze(), i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test and visualize a trained ContinuousThoughtMachineGridDiff model.")
    
    # Paths
    parser.add_argument('--model_path', type=str, 
                        default='C:\\\\Users\\\\abhay\\\\OneDrive\\\\IIITD\\\\ARC\\\\continuous-thought-machines\\\\logs\\\\scratch\\\\checkpoint.pt',
                        help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--dataset_path', type=str, 
                        default='C:\\\\Users\\\\abhay\\\\OneDrive\\\\IIITD\\\\ARC\\\\continuous-thought-machines\\\\data\\\\grid_diff\\\\grid_diff_dataset.json',
                        help='Path to the grid_diff_dataset.json file.')

    # Model Hyperparameters (should match training)
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input for CTM.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks/iterations.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes (should be 2 for same/different).')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='none', help='Type of backbone featureiser used (e.g., none, resnet18-4).')

    # Visualization and Data Loading
    parser.add_argument('--num_samples_visualize', type=int, default=5, help='Number of test samples to visualize.')
    parser.add_argument('--test_split_ratio', type=float, default=0.2, help='Proportion of dataset for testing (should match data preparation).')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available.')
    
    args = parser.parse_args()
    main(args)
