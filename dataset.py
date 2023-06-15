import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np
import random

from label.one_word.utils import phoneme_dict
from language_utils import list_of_phoneme, vocab_of_word
from config import load_arg


class Dataset(Dataset):
    def __init__(self, args, mode="train") -> None:
        super().__init__()

        # 数据集所在的文件夹
        self.data_root = args.data_root

        # 声母、韵母序列
        self.phoneme = ["<pad>", ""] + list_of_phoneme
        self.phoneme_to_idx = {}
        self.phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(self.phoneme)}
        self.idx_to_phoneme = {}
        self.idx_to_phoneme = {idx: phoneme for idx, phoneme in enumerate(self.phoneme)}

        # 加载 pkl 文件
        pkl_file = args.train_pkl_file if mode == "train" else args.test_pkl_file
        with open(pkl_file, "rb") as file:
            data = pickle.load(file)

        self.file_lst = [file for file in data.keys() if data[file][1] is not None]
        self.boundaries = [
            data[file][1] for file in data.keys() if data[file][1] is not None
        ]  # [(0, 4, 'zai'), (4, 9, 'guo'), (9, 12, 'qu'), (12, 13, 'de'), (13, 16, 'yi'), (16, 21, 'nian'), (21, 26, 'li')]

        # 添加负样本
        self._add_negative()

    def _add_negative(self):
        # 添加负样本
        for idx in range(len(self.file_lst)):
            positive_pinyin = [pinyin for start, end, pinyin in self.boundaries[idx]]
            diff_pinyin = list(set(vocab_of_word) - set(positive_pinyin))
            negative_pinyin = random.sample(diff_pinyin, len(positive_pinyin))
            for n_pinyin in negative_pinyin:
                # 边界设置为(-1, -1) 表明这是一个负样本
                self.boundaries[idx].append((-1, -1, n_pinyin))

    def _get_video_features(self, idx):
        # file_name = self.file_lst[idx].replace("/", "\")
        video_features = np.load("{}/{}.npy".format(self.data_root, self.file_lst[idx]))
        video_features = torch.FloatTensor(video_features)

        return video_features

    def _get_key_word_features(self, idx):
        # shape = (num_word_in_video, num_phoneme_lst)
        key_word_features = []

        for start, end, pinyin in self.boundaries[idx]:
            idx_lst = [self.phoneme_to_idx[phoneme] for phoneme in phoneme_dict(pinyin)]
            key_word_features.append(idx_lst)
            # self.key_word_features.append(tmp_lst)
            # idx_lst.append(self.phoneme_to_idx[phoneme])
        return key_word_features

    def _get_label(self, idx):
        detection_label = []
        location_label = []

        # 视频的持续时间，单位是帧
        duration = self._get_video_features(idx).shape[0]
        for i, (start, end, pinyin) in enumerate(self.boundaries[idx]):
            word_index = vocab_of_word.index(pinyin)
            if start == -1 and end == -1:
                # 如果是负样本
                detection_label.append(torch.tensor(0))
                location_label.append(torch.zeros(duration))

            else:
                # 如果是正样本
                detection_label.append(torch.tensor(1))
                lst = torch.zeros(duration)
                lst[start : end + 1] = 1
                location_label.append(lst)

        detection_label = torch.stack(detection_label)
        location_label = torch.stack(location_label)

        return detection_label, location_label

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, idx):
        detection_label, location_label = self._get_label(idx)

        video_features = self._get_video_features(idx)
        video_features = video_features.repeat(len(detection_label), 1, 1)

        key_word_features = self._get_key_word_features(idx)

        print(len(key_word_features))
        print(len(key_word_features[0]))

        return (
            video_features,
            self._get_key_word_features(idx),
            detection_label,
            location_label,
        )


def pading_text(texts):
    # 最终的两个输出
    texts_paded = []
    texts_mask = []
    # 找出当前 texts 的最长长度
    max_len = max([len(text) for text in texts])
    # 遍历 texts，补0并生成对应的掩码
    for text in texts:
        if text.sixe(0) == max_len:
            texts_paded.append(text)
            texts_mask.append(torch.ones(len(text)))
        else:
            texts_paded.append(
                torch.cat([text, torch.zeros(max_len - len(text))], dim=0)
            )
            texts_mask.append(
                torch.cat([torch.ones(len(text)), torch.zeros(max_len - len(text))])
            )

    texts_paded = torch.stack(texts_paded, dim=0).long()
    texts_mask = torch.stack(texts_mask, dim=0).bool().unsqueeze(1)

    return texts_paded, texts_mask


def padding(texts):
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    mask_texts = padded_texts != 0

    return padded_texts, mask_texts


def test():
    args = load_arg()
    dataset = Dataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for features, text, label_1, label_2 in dataloader:
        print("*" * 100)
        print(features.shape)
        print(type(text))
        print(label_1.shape)
        print(label_2.shape)
        break


if __name__ == "__main__":
    test()
