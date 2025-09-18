# This file is built upon the official RoNIN implementation 
# https://github.com/Sachini/ronin/tree/master

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn


def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_planes, out_planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = conv3(out_planes, out_planes, kernel_size, stride, dilation)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.conv3 = nn.Conv1d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FCOutputModule(nn.Module):
    """
    Fully connected output module.
    """
    def __init__(self, in_planes, num_outputs, **kwargs):
        """
        Constructor for a fully connected output layer.

        Args:
          in_planes: number of planes (channels) of the layer immediately proceeding the output module.
          num_outputs: number of output predictions.
          fc_dim: dimension of the fully connected layer.
          dropout: the keep probability of the dropout layer
          trans_planes: (optional) number of planes of the transition convolutional layer.
        """
        super(FCOutputModule, self).__init__()
        fc_dim = kwargs.get('fc_dim', 1024)
        dropout = kwargs.get('dropout', 0.5)
        in_dim = kwargs.get('in_dim', 7)
        trans_planes = kwargs.get('trans_planes', None)
        if trans_planes is not None:
            self.transition = nn.Sequential(
                nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(trans_planes))
            in_planes = trans_planes
        else:
            self.transition = None

        self.fc = nn.Sequential(
            nn.Linear(in_planes * in_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_outputs))

    def get_dropout(self):
        return [m for m in self.fc if isinstance(m, torch.nn.Dropout)]

    def forward(self, x):
        if self.transition is not None:
            x = self.transition(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


class GlobAvgOutputModule(nn.Module):
    """
    Global average output module.
    """
    def __init__(self, in_planes, num_outputs):
        super(GlobAvgOutputModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_planes, num_outputs)

    def get_dropout(self):
        return []

    def forward(self, x):
        x = self.avg()
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet1D(nn.Module):
    def __init__(self, num_inputs, num_outputs, block_type, group_sizes, base_plane=64, output_block=None,
                 zero_init_residual=False, **kwargs):
        super(ResNet1D, self).__init__()
        self.base_plane = base_plane
        self.inplanes = self.base_plane

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(num_inputs, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual groups
        self.planes = [self.base_plane * (2 ** i) for i in range(len(group_sizes))]
        kernel_size = kwargs.get('kernel_size', 3)
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [self._make_residual_group1d(block_type, self.planes[i], kernel_size, group_sizes[i],
                                              strides[i], dilations[i])
                  for i in range(len(group_sizes))]
        self.residual_groups = nn.Sequential(*groups)

        # Output module
        if output_block is None:
            self.output_block = GlobAvgOutputModule(self.planes[-1] * block_type.expansion, num_outputs)
        else:
            self.output_block = output_block(self.planes[-1] * block_type.expansion, num_outputs, **kwargs)

        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block_type, planes, kernel_size, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block_type.expansion))
        layers = []
        layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size,
                                 stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = self.output_block(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def load_dataset(data_path, window_size=200, stride=10):
    ts_column = ["time(us)"]
    source_columns = ["gx(rad/s)", "gy(rad/s)", "gz(rad/s)", "ax(m/s^2)", "ay(m/s^2)", "az(m/s^2)"]
    target_columns = ["px", "py", "pz"]
    
    source_sequences = []
    target_sequences = []
    target_velocities = []
    
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
        
            # Read the file using pandas
            df = pd.read_csv(file_path, sep=" ")
            
            # Extract required columns
            source_data = df[source_columns]
            target_data = df[target_columns]
            ts_data = df[ts_column]
        
            # Create source sequences of size (100, 6)
            for i in range(0, len(source_data) - window_size, stride):
                source_seq = source_data.iloc[i:i+window_size, :].values
                source_sequences.append(source_seq)
        
            # Create target sequences of size (100, 3)
            for i in range(0, len(target_data) - window_size, stride):
                target_seq = target_data.iloc[i:i+window_size, :].values
                target_sequences.append(target_seq)
                ts_seq = ts_data.iloc[i:i+window_size, :].values
                velocity = (target_seq[-1,:] - target_seq[0,:]) / ((ts_seq[-1,:] - ts_seq[0,:])/1e6)  # convert microseconds to seconds
                target_velocities.append(velocity)                
        
            # Replace the last source sequence with the last window_size entries of the file
            last_source_seq = source_data.iloc[-window_size:, :].values
            source_sequences[-1] = last_source_seq
    
            # Replace the last target sequence with the last window_size entries of the file
            last_target_seq = target_data.iloc[-window_size:, :].values
            target_sequences[-1] = last_target_seq

            last_ts_seq = ts_data.iloc[-window_size:,:].values
            last_target_velocity = (last_target_seq[-1,:] - last_target_seq[0,:]) / ((last_ts_seq[-1,:] - last_ts_seq[0,:])/1e6)
            target_velocities[-1] = last_target_velocity
    
    # Subtract the first row from all rows in each target sequence
    target_sequences = [seq - seq[0] for seq in target_sequences]
    
    # Convert source sequences to torch tensor
    source_tensors = torch.stack([torch.from_numpy(seq) for seq in source_sequences]).to(torch.float32)
    
    # Convert target sequences to torch tensor
    target_pos_tensors = torch.stack([torch.from_numpy(seq) for seq in target_sequences]).to(torch.float32)
    target_vel_tensors = torch.stack([torch.from_numpy(seq) for seq in target_velocities]).to(torch.float32)
    
    # Create TensorDatasets
    dataset = TensorDataset(source_tensors, target_vel_tensors, target_pos_tensors)

    return dataset


def train_loop(model, opt, loss_fn, dataloader, device=torch.device('cpu')):

    model.train()
    train_loss_list = list()

    for batch in dataloader:
        inputs, targets, _ = batch
        inputs = inputs.permute(0,2,1).to(device)
        targets = targets.to(device)

        opt.zero_grad()

        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt.step()

        train_loss_list.append(loss.detach().cpu().numpy())
        
    return np.mean(train_loss_list)


def validation_loop(model, loss_fn, dataloader, device=torch.device('cpu')):

    model.eval()
    val_loss_list = list()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, _ = batch
            inputs = inputs.permute(0,2,1).to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = loss_fn(preds, targets)  

            val_loss_list.append(loss.detach().cpu().numpy())          

        
    return np.mean(val_loss_list)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device=torch.device('cpu'), scheduler=None, model_name='ronin_resnet'):
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    best_val_loss = np.inf
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device=device)
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
    parser.add_argument("--model_name", type=str, default="ronin_resnet")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--window_size", type=int, default=120) # mocap has 120 Hz
    parser.add_argument("--stride", type=int, default=10) # mocap has 120 Hz
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--trans_planes", type=int, default=128)

    args = parser.parse_args()

    train_data_path = './../dat/merged/train'
    train_dataset = load_dataset(train_data_path, window_size=args.window_size, stride=args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_data_path = './../dat/merged/val'
    val_dataset = load_dataset(val_data_path, window_size=args.window_size, stride=args.stride)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    _fc_config = {'fc_dim': args.fc_dim, 'in_dim': (args.window_size // 32 + 1), 'dropout': args.dropout, 'trans_planes': args.trans_planes}
    model = ResNet1D(6, 3, BasicBlock1D, [2, 2, 2, 2],
                        base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10, verbose=True, eps=1e-12)    
    loss_fn = nn.MSELoss()

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_loader, val_loader, args.epochs, 
                                                device=device, scheduler=scheduler, model_name=args.model_name)

    with open(f'./../jobs/{args.model_name}_train_loss_list.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)

    with open(f'./../jobs/{args.model_name}_val_loss_list.pkl', 'wb') as f:
        pickle.dump(validation_loss_list, f)

    torch.save(model, f"./../jobs/{args.model_name}_last.pt")


if __name__=="__main__":
    main()