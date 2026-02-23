import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Convolutional Vision Transformer (CViT) implementation.
        :param num_classes: Number of output classes for classification.
        :param embed_dim: Dimensionality of the embedding space.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
        :param patch_size: Size of the convolutional patches.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param image_size: Height/width of the square input image.
        """
        super(Model, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2  # Total number of patches after conv
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the CViT model.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of shape (B, num_classes)
        """
        B = x.size(0)
        x = self.conv1(x)                  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(start_dim=1)         # (B, embed_dim * num_patches)
        x = self.linear_proj(x)            # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])        # Use [CLS] token for classification

# === Test config ===
batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]
