import torch
import torch.nn as nn

class CTMGridWrapper(nn.Module):
    """
    Wraps a CTM model for grid-to-grid prediction (ARC/maze style).
    Predicts output grid dimensions, then fills the grid cell by cell.
    """
    def __init__(self, ctm_model, max_grid_size=30, num_colors=10):
        super().__init__()
        self.ctm = ctm_model
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        # Heads for predicting height and width (classification)
        self.size_head = nn.Linear(ctm_model.d_model, max_grid_size * 2)  # [height, width]
        # Head for predicting cell value
        self.cell_head = nn.Linear(ctm_model.d_model, num_colors)

    def forward(self, input_grid, partial_output_grid, position):
        """
        input_grid: [B, H_in, W_in] (int)
        partial_output_grid: [B, H_out, W_out] (int, filled so far, 0 for empty)
        position: [B, 2] (row, col) of the cell to predict
        """
        B = input_grid.size(0)
        H_in, W_in = input_grid.size(1), input_grid.size(2)
        H_out, W_out = partial_output_grid.size(1), partial_output_grid.size(2)
        # Convert to float for model
        input_flat = input_grid.float().view(B, -1)
        output_flat = partial_output_grid.float().view(B, -1)
        pos_embed = position.float() / self.max_grid_size  # normalize
        x = torch.cat([input_flat, output_flat, pos_embed], dim=1)
        x = x.unsqueeze(1)  # [B, 1, D]
        ctm_out, _, _ = self.ctm(x)  # [B, d_model, iterations]
        ctm_out = ctm_out[:, :, -1]  # [B, d_model]
        cell_logits = self.cell_head(ctm_out)  # [B, num_colors]
        return cell_logits

    def predict_grid_size(self, input_grid):
        """Predict output grid height and width from input grid."""
        B = input_grid.size(0)
        input_flat = input_grid.float().view(B, -1)
        x = input_flat.unsqueeze(1)
        ctm_out, _, _ = self.ctm(x)
        ctm_out = ctm_out[:, :, -1]
        size_logits = self.size_head(ctm_out)  # [B, max_grid_size*2]
        height_logits = size_logits[:, :self.max_grid_size]
        width_logits = size_logits[:, self.max_grid_size:]
        return height_logits, width_logits
