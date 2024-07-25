import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import os
import pandas as pd
import argparse
import pickle
import math
import os

def load_dataset(data_path, window_size=120, stride=10):
    ts_column = ["time(us)"]
    source_columns = ["gx(rad/s)", "gy(rad/s)", "gz(rad/s)", "ax(m/s^2)", "ay(m/s^2)", "az(m/s^2)"]
    target_columns = ["px", "py", "pz"]
    
    source_sequences = []
    target_sequences = []
    input_velocities = []

    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
        
            # Read the file using pandas
            df = pd.read_csv(file_path, sep=" ")    
            
            # Extract required columns
            source_data = df[source_columns]
            target_data = df[target_columns]
            ts_data = df[ts_column]
        
            # Create source sequences
            for i in range(0, len(source_data) - window_size, stride):
                source_seq = source_data.iloc[i:i+window_size, :].values
                source_sequences.append(source_seq)
        
            # Create target sequences
            for i in range(0, len(target_data) - window_size, stride):
                target_seq = target_data.iloc[i:i+window_size, :].values
                target_sequences.append(target_seq)
                # calculate v0
                if i==0:
                    input_velocities.append(np.zeros(3))
                else:
                    time_delta = ((ts_data.iloc[i] - ts_data.iloc[i-1])/1e6).values
                    pos_delta = target_data.iloc[i,:].values - target_data.iloc[i-1, :].values
                    v0 = pos_delta/time_delta
                    input_velocities.append(v0)

            # Replace the last source sequence with the last window_size entries of the file
            last_source_seq = source_data.iloc[-window_size:, :].values
            source_sequences[-1] = last_source_seq
    
            # Replace the last target sequence with the last window_size entries of the file
            last_target_seq = target_data.iloc[-window_size:, :].values
            target_sequences[-1] = last_target_seq

            # Replace last v0 with the v0 for the corrected last window from above
            last_time_delta = ((ts_data.iloc[-100] - ts_data.iloc[-101])/1e6).values
            last_pos_delta = target_data.iloc[-100,:].values - target_data.iloc[-101, :].values
            last_input_velocity = last_pos_delta / last_time_delta
            input_velocities[-1] = last_input_velocity
    
    # Subtract the first row from all rows in each target sequence
    target_sequences = [seq - seq[0] for seq in target_sequences]    

    # Convert source sequences to torch tensor
    source_tensors = torch.stack([torch.from_numpy(seq) for seq in source_sequences]).to(torch.float32)
    v0_tensors = torch.stack([torch.from_numpy(seq) for seq in input_velocities]).to(torch.float32)
    
    # Convert target sequences to torch tensor
    target_pos_tensors = torch.stack([torch.from_numpy(seq) for seq in target_sequences]).to(torch.float32)
    
    # Create TensorDatasets
    dataset = TensorDataset(source_tensors, v0_tensors, target_pos_tensors)

    return dataset, df


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class TransformerClassifierModel(nn.Module):

    def __init__(self, d_model: int, input_dim: int, transformer_output_dim: int, final_output_dim: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, seq_len=120, device=torch.device('cpu')):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding_replacement = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, transformer_output_dim)
        self.head = nn.Sequential(
            nn.Linear(transformer_output_dim, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, final_output_dim)
        )
        self.device=device

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # src = torch.cat([(-1)*torch.ones((1, src.size(1), src.size(2))).to(self.device), src], dim=0)
        src = self.embedding_replacement(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        output = output.permute(1,0,2)
        output = self.head(output)
        return output
    
def perform_sins(acc, dt, v_0=torch.zeros(3), device=torch.device('cpu')):
    """
        acc - torch.tensor(seq_len, 3): acceleration values
        dt - float: time between two measurements (mocap: 1/120)
    """

    # Initialize velocity and positions
    v_0 = v_0.to(device)
    v_arr = torch.cumsum(acc[1:]*dt, dim=0) + v_0
    acc_arr = 0.5 * acc[1:] * (dt**2)
    p_arr = torch.vstack([acc[0]*0, torch.cumsum(v_arr*dt+acc_arr, dim=0)])
    return p_arr


def train_loop(model, opt, loss_fn, dataloader, device=torch.device('cpu'), clip_grad_norm=1):

    model.train()
    train_loss_list = list()

    for batch in dataloader:
        inputs, v0, targets = batch
        inputs = torch.cat([torch.cat([v0,v0], dim=1).unsqueeze(1), inputs], dim=1)  # prepend 
        inputs = inputs.permute(1,0,2).to(device)
        targets = targets.to(device)

        opt.zero_grad()

        preds = model(inputs)
        preds_integrated = torch.stack([perform_sins(preds[batch_idx,1:,:], 1/120, device=device)
                                      for batch_idx in range(preds.size(0))], dim=0)
        loss = loss_fn(preds_integrated[:,10:-10,:], targets[:,10:-10,:])  # take only middle values for context
        loss.backward()
        opt.step()

        train_loss_list.append(loss.detach().cpu().numpy())
        
    return np.mean(train_loss_list)


def validation_loop(model, loss_fn, dataloader, device=torch.device('cpu')):
    
    model.eval()
    val_loss_list = list()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, v0, targets = batch
            inputs = torch.cat([torch.cat([v0,v0], dim=1).unsqueeze(1), inputs], dim=1)  # prepend 
            inputs = inputs.permute(1,0,2).to(device)
            targets = targets.to(device)

            preds = model(inputs)
            preds_integrated = torch.stack([perform_sins(preds[batch_idx,1:,:], 1/120, device=device)
                                        for batch_idx in range(preds.size(0))], dim=0)            
            loss = loss_fn(preds_integrated[:,10:-10,:], targets[:,10:-10,:])  # take only middle values for context

            val_loss_list.append(loss.detach().cpu().numpy())          
        
    return np.mean(val_loss_list)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device=torch.device('cpu'), clip_grad_norm=1, scheduler=None, model_name="denoise"):
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    best_val_loss = np.inf    
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device=device, clip_grad_norm=clip_grad_norm)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader, device=device)
        validation_loss_list += [validation_loss]

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model, f"./../jobs/{model_name}_best.pt")

        if scheduler:
            scheduler.step(metrics=validation_loss)
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="denoise")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--window_size", type=int, default=120) # mocap has 120 Hz
    parser.add_argument("--stride", type=int, default=10) # mocap has 120 Hz
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument("--d_model", type=int, default=200)    
    parser.add_argument("--d_hid", type=int, default=128) 
    parser.add_argument("--transformer_output_dim", type=int, default=512) 
    parser.add_argument("--n_layers", type=int, default=2) 
    parser.add_argument("--n_heads", type=int, default=4) 
    args = parser.parse_args()

    train_data_path = './../dat/merged/train'
    train_dataset, train_df = load_dataset(train_data_path, window_size=args.window_size, stride=args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_data_path = './../dat/merged/val'
    val_dataset, val_df = load_dataset(val_data_path, window_size=args.window_size, stride=args.stride)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = TransformerClassifierModel(d_model=args.d_model, input_dim=6, transformer_output_dim=args.transformer_output_dim, final_output_dim=3, nhead=args.n_heads, 
                                       d_hid=args.d_hid, nlayers=args.n_layers, dropout=0.2, seq_len=args.window_size, device=device)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=40, verbose=True, eps=1e-12)    
    loss_fn = nn.L1Loss()

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_loader, val_loader, args.epochs, 
                                                device=device, scheduler=scheduler, model_name=args.model_name)

    with open(f'./../jobs/{args.model_name}_train_loss_list.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)

    with open(f'./../jobs/{args.model_name}_val_loss_list.pkl', 'wb') as f:
        pickle.dump(validation_loss_list, f)

    torch.save(model, f"./../jobs/{args.model_name}_last.pt")

if __name__=="__main__":
    main()