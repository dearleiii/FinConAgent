import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def soft_threshold(x, lam):
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam / 2.0, min=0.0)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=3, input_shape=(1, 25, 8)):
        super().__init__()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        
        self.latent_dim = latent_dim
        self.input_shape = input_shape  # (channels, seq_len, num_features)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        
        # Calculate encoder output size dynamically
        self.encoder_out_size = self._calculate_encoder_size(input_shape)
        
        self.fc_enc = nn.Linear(self.encoder_out_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.encoder_out_size)
        
        # Store decoder reshape dimensions
        self.decoder_reshape = self._get_decoder_shape()

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def _calculate_encoder_size(self, input_shape):
        """Calculate encoder output size for flattening"""
        # Create dummy input to compute output size
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = self.encoder(dummy_input)
        return dummy_output.view(1, -1).shape[1]
    
    def _get_decoder_shape(self):
        """Get the shape needed for decoder input"""
        dummy_input = torch.zeros(1, *self.input_shape)
        dummy_output = self.encoder(dummy_input)
        return tuple(dummy_output.shape[1:])  # (channels, height, width)

    def forward(self, x):
        # x shape: [batch_size, 1, seq_len, features]
        input_shape = x.shape
        batch_size = x.size(0)
        
        z = self.encoder(x)
        z = z.view(batch_size, -1)
        z = self.fc_enc(z)

        y = self.fc_dec(z)
        y = y.view(batch_size, *self.decoder_reshape)
        out = self.decoder(y)
        
        return out
