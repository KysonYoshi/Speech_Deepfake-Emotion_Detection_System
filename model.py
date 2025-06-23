import torch.nn as nn
import fairseq
from dataclasses import dataclass, field
import torch.nn.functional as F
import torch
from mamba_blocks import MixerModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    
class SSLModel(nn.Module): #W2V
    def __init__(self,device):
        super(SSLModel, self).__init__()
        cp_path = './xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + mamba')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.config = MambaConfig(d_model=args.emb_size, n_layer=args.num_encoders // 2)
        print(self.config)
        self.conformer=MixerModel(d_model=self.config.d_model, 
                                  n_layer=self.config.n_layer, 
                                  ssm_cfg=self.config.ssm_cfg, 
                                  rms_norm=self.config.rms_norm, 
                                  residual_in_fp32=self.config.residual_in_fp32, 
                                  fused_add_norm=self.config.fused_add_norm
                                  )
    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat,_ = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out =self.conformer(x) 
        return out

class MambaHead(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + mamba')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.config = MambaConfig(d_model=args.emb_size, n_layer=args.num_encoders // 2)
        print(self.config)
        self.conformer=MixerModel(d_model=self.config.d_model, 
                                  n_layer=self.config.n_layer, 
                                  ssm_cfg=self.config.ssm_cfg, 
                                  rms_norm=self.config.rms_norm, 
                                  residual_in_fp32=self.config.residual_in_fp32, 
                                  fused_add_norm=self.config.fused_add_norm
                                  )
    def forward(self, x_ssl_feat):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        #x_ssl_feat,_ = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out =self.conformer(x) 
        return out

class EmotionClassifier(nn.Module):
    def __init__(
        self,
        ssl: SSLModel,
        feat_dim: int = 1024,
        proj_dim: int = 64,
        num_classes: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.5
    ):
        super().__init__()
        self.ssl = ssl

        # learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        # MLP head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, embs: torch.Tensor, lengths: list[int] | None = None) -> torch.Tensor:
        """
        Args:
            embs: Tensor of shape (B, T, D)
            lengths: Optional list of true sequence lengths (<= T) for each batch entry
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        B, T, D = embs.size()
        device = embs.device

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls_tokens, embs], dim=1)               # (B, T+1, D)
        new_T = T + 1

        # build padding mask for transformer
        if lengths is not None:
            lengths_tensor = torch.tensor(lengths, device=device)
            arange = torch.arange(new_T, device=device).unsqueeze(0)  # (1, T+1)
            # CLS token at pos=0 always valid
            valid_cls = arange == 0
            # original tokens shifted by +1
            valid_tokens = (arange - 1) < lengths_tensor.unsqueeze(1)
            valid_mask = valid_cls | valid_tokens                # (B, T+1)
            key_padding_mask = ~valid_mask                       # (B, T+1)
        else:
            key_padding_mask = None

        # Transformer expects (S, N, E)
        x = x.permute(1, 0, 2)                                   # (T+1, B, D)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = x.permute(1, 0, 2)                                   # (B, T+1, D)

        # take CLS output for classification
        cls_output = x[:, 0, :]                                  # (B, D)
        proj = self.projector(cls_output)                       # (B, proj_dim)
        logits = self.classifier(proj)                          # (B, num_classes)
        return logits, cls_output
