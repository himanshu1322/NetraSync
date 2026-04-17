import torch
import torch.nn as nn

class NetraSyncFusion(nn.Module):
    def __init__(self, embed_dim=64): # Increased to 64 for higher capacity
        super(NetraSyncFusion, self).__init__()
        
        self.gaze_embedder = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        self.eeg_embedder = nn.Sequential(
            nn.Linear(8, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), # More neurons
            nn.ReLU(),
            nn.Dropout(0.1),         # Prevents overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, gaze, eeg):
        g_feat = self.gaze_embedder(gaze).unsqueeze(1) 
        e_feat = self.eeg_embedder(eeg)                
        attn_output, _ = self.cross_attention(g_feat, e_feat, e_feat)
        x = attn_output.squeeze(1) 
        return self.classifier(x), attn_output