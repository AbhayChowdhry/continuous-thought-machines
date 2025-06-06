"""
Train a CTM model on ARC training data and evaluate its performance.
This script is intended as a baseline for the ARC task.
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.ctm import ContinuousThoughtMachine
from arc.ctm_sequence_wrapper import CTMGridWrapper

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ARC Dataset
class ARCDataset(Dataset):
    def __init__(self, data_dir, max_tasks=None, max_grid_size=30):
        self.data_dir = Path(data_dir) / "training"
        self.task_files = list(self.data_dir.glob("*.json"))
        if max_tasks:
            self.task_files = self.task_files[:max_tasks]
        self.max_grid_size = max_grid_size

    def __len__(self):
        return len(self.task_files)

    def __getitem__(self, idx):
        with open(self.task_files[idx], 'r') as f:
            task_data = json.load(f)
        # Use only the first test example for simplicity
        test_example = task_data['test'][0]
        input_grid = torch.tensor(test_example['input'], dtype=torch.long)
        output_grid = torch.tensor(test_example['output'], dtype=torch.long)
        return input_grid, output_grid

def pad_grid(grid, target_h, target_w):
    h, w = grid.shape
    padded = torch.zeros((target_h, target_w), dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded

def custom_collate(batch):
    input_grids, output_grids = zip(*batch)
    max_h_in = max(g.shape[0] for g in input_grids)
    max_w_in = max(g.shape[1] for g in input_grids)
    max_h_out = max(g.shape[0] for g in output_grids)
    max_w_out = max(g.shape[1] for g in output_grids)
    input_grids_padded = [pad_grid(g, max_h_in, max_w_in) for g in input_grids]
    output_grids_padded = [pad_grid(g, max_h_out, max_w_out) for g in output_grids]
    input_batch = torch.stack(input_grids_padded)
    output_batch = torch.stack(output_grids_padded)
    return input_batch, output_batch

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc_data_dir = r"c:\Users\abhay\OneDrive\IIITD\ARC\ARC-AGI-master\ARC-AGI-master\data"
    max_grid_size = 30
    batch_size = 2
    dataset = ARCDataset(arc_data_dir, max_tasks=50, max_grid_size=max_grid_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=custom_collate)

    base_model = ContinuousThoughtMachine(
        iterations=2,
        d_model=128,
        d_input=128,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        memory_length=4,
        deep_nlms=True,
        memory_hidden_dims=64,
        do_layernorm_nlm=False,
        backbone_type='none',
        positional_embedding_type='none',
        out_dims=128,
        prediction_reshaper=[-1],
        dropout=0.1,
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    ).to(device)
    model = CTMGridWrapper(base_model, max_grid_size=max_grid_size, num_colors=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 1
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_grid, output_grid in pbar:
            input_grid = input_grid.to(device)  # [B, H_in, W_in]
            output_grid = output_grid.to(device)  # [B, H_out, W_out]
            B, H_out, W_out = output_grid.shape
            # Predict grid size
            height_logits, width_logits = model.predict_grid_size(input_grid)
            true_height = torch.tensor([H_out]*B, dtype=torch.long, device=device)
            true_width = torch.tensor([W_out]*B, dtype=torch.long, device=device)
            size_loss = criterion(height_logits, true_height) + criterion(width_logits, true_width)
            # Predict each cell (teacher forcing)
            cell_loss = 0.0
            for i in range(H_out):
                for j in range(W_out):
                    # Prepare partial output grid (mask future cells as 0)
                    partial_output = output_grid.clone()
                    partial_output[:, i:, j:] = 0
                    pos = torch.tensor([[i, j]]*B, dtype=torch.long, device=device)
                    logits = model(input_grid, partial_output, pos)  # [B, num_colors]
                    target = output_grid[:, i, j]
                    cell_loss += criterion(logits, target)
            loss = size_loss + cell_loss / (H_out * W_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch+1} avg loss: {epoch_loss/len(dataloader):.4f}")
        torch.cuda.empty_cache()
    # Save model
    save_path = "arc/ctm_arc_model.pth"
    os.makedirs("arc", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Evaluation after training
    print("\nEvaluating model on training data...")
    model.eval()
    with torch.no_grad():
        for idx in range(min(5, len(dataset))):  # Show 5 examples
            input_grid, target_grid = dataset[idx]
            input_grid = input_grid.unsqueeze(0).to(device)  # [1, H_in, W_in]
            target_grid = target_grid.unsqueeze(0).to(device)  # [1, H_out, W_out]
            # Predict grid size
            height_logits, width_logits = model.predict_grid_size(input_grid)
            pred_height = height_logits.argmax(dim=1).item()
            pred_width = width_logits.argmax(dim=1).item()
            pred_height = max(1, min(pred_height, max_grid_size))
            pred_width = max(1, min(pred_width, max_grid_size))
            # Initialize output grid
            output_grid = torch.zeros((1, pred_height, pred_width), dtype=torch.long, device=device)
            for i in range(pred_height):
                for j in range(pred_width):
                    pos = torch.tensor([[i, j]], dtype=torch.long, device=device)
                    logits = model(input_grid, output_grid, pos)  # [1, num_colors]
                    pred_cell = logits.argmax(dim=1)
                    output_grid[:, i, j] = pred_cell
            predicted_grid = output_grid.squeeze(0).cpu().tolist()
            input_grid_np = input_grid.squeeze(0).cpu().tolist()
            target_grid_np = target_grid.squeeze(0).cpu().tolist()
            # Print results
            print(f"\nExample {idx+1}")
            print("Test input grid:")
            for row in input_grid_np:
                print(' '.join(str(cell) for cell in row))
            print("Ground truth output grid:")
            for row in target_grid_np:
                print(' '.join(str(cell) for cell in row))
            print("Predicted output grid:")
            for row in predicted_grid:
                print(' '.join(str(cell) for cell in row))

if __name__ == "__main__":
    main()
