import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from config import load_arg


# 构建数据集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.lst = torch.rand(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return [self.data[index], self.lst[index]]


# 定义collate_fn函数
def my_collate(batch):
    # print(type(batch))
    features = [b[0] for b in batch]  # (batch_size, T, 512)
    feature_T = [len(feature) for feature in features]  # [30, 50]
    lst = [b[1] for b in batch]

    features = pad_sequence(
        features, batch_first=True
    )  # 将数据填充到相同的长度: (batch_size, max_T, 512)
    # print(features.shape)

    lst = torch.stack(lst)

    lst += 1

    features_mask = torch.zeros((features.shape[0], features.shape[1]))
    for i in range(len(features)):
        features_mask[i, feature_T[i] :] = 1
    # features_mask = features_mask.bool()

    return features, features_mask, lst


# 创建数据集和DataLoader对象
data = [
    torch.randn(50, 512),
    torch.randn(30, 512),
    torch.randn(70, 512),
    torch.randn(80, 512),
]
dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=my_collate)


# 定义Transformer模型
"""
class TransformerModel(torch.nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers,
        )
        self.fc = torch.nn.Linear(d_model, 10)

    def forward(self, x, lengths):
        # 将数据转换为pack格式，并进行Transformer Encoder计算
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        x = self.transformer_encoder(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # 全连接层计算
        x = self.fc(x[:, -1, :])  # 取最后一个时刻的输出向量

        return x

"""

args = load_arg()
video_encoder = TransformerEncoder(
    TransformerEncoderLayer(
        d_model=args.d_model,
        nhead=args.nhead,
        dropout=args.dropout,
        batch_first=args.batch_first,
    ),
    num_layers=args.num_layers,
)
# 测试模型
# model = TransformerModel()
for features, features_mask, lst in dataloader:
    print("*" * 100)
    print(features.shape)
    print(features_mask)
    print(lst.shape)
    # output = video_encoder(batch, lengths)
    # print(output.shape)
