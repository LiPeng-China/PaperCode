import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math


class Kws_Net(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.video_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.nhead,
                dropout=args.dropout,
                batch_first=args.batch_first,
            ),
            num_layers=args.num_layers,
        )
        self.key_word_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.nhead,
                dropout=args.dropout,
                batch_first=args.batch_first,
            ),
            num_layers=args.num_layers,
        )

        self.multimodal_fusion = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.nhead,
                dropout=args.dropout,
                batch_first=args.batch_first,
            ),
            num_layers=args.num_layers,
        )

        self.position_encoder = PositionEncoder(
            d_model=args.d_model, dropout=args.dropout
        )
        self.text_embeding = nn.Embedding(args.vocab, args.d_model)

        self.detector = nn.Linear(args.d_model, 1)
        self.locator = nn.Linear(args.d_model, 1)

        self.cls = nn.Parameter(torch.randn(1, 1, args.d_model))

    def forward(self, video, video_mask, key_word, key_word_mask):
        video = self.position_encoder(video)
        video = self.video_encoder(video, video_mask)

        # 如果是一条视频对应 N 个关键词时，将 video 及其掩码 复制 N 份
        if video.size(0) == 1 and video.size(0) != key_word.size(1):
            video = video.repeat(key_word.size(0), 1, 1)
            video_mask = video_mask.repeat(key_word.size(0), 1, 1)

        key_word = self.text_embeding(key_word)
        key_word = self.position_encoder(key_word)
        key_word = self.key_word_encoder(key_word, key_word_mask)

        token = torch.cat(
            [self.cls.repeat(video.size(0), 1, 1), video, key_word], dim=1
        )

        if video_mask is None or key_word_mask is None:
            token_mask = None
        else:
            token_mask = torch.cat(
                [video_mask[..., 0], video_mask, key_word_mask], dim=2
            )

        fusion_out = self.multimodal_fusion(token, token_mask)

        detection = self.detector(fusion_out[:, 0])
        location = self.locator(fusion_out[:, 1 : video.size(1) + 1])
        return detection, location


class PositionEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def init_model(args):
    model = Kws_Net(args)

    # 参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)

encoder = TransformerEncoder(encoder_layer, num_layers=6)

ipt = torch.rand((2, 40, 512))
ipt_mask = torch.zeros_like(ipt)
ipt_mask[:, :33, :] = 1
# ipt_mask = torch.rand((1, 40, 512))
# out = encoder(ipt, ipt_mask)
# padding_mask = torch.ones(2, 40, dtype=torch.bool)
# padding_mask[1, 3:] = False
out1 = encoder(torch.rand((2, 40, 512)))
out2 = encoder(torch.rand((2, 70, 512)))
print(out1.shape)
print(out2.shape)
# from config import load_arg

# args = load_arg()
# model = Kws_Net(args)
# print(model)
# print(encoder)
