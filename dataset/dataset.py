import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

import sys
sys.path.append(r"/mnt/c/Users/peng/Code/PaperCode/dicts")
from list_of_phoneme import list_of_phoneme
from list_of_pinyin import list_of_pinyin

sys.path.append(r"/mnt/c/Users/peng/Code/PaperCode/utils")
from utils import pinyins_to_phonemes


class MyDataset(Dataset):
    def __init__(self, args, mode='train') -> None:
        super().__init__()

        #* 1、加载 csv 文件中的内容
        if mode == 'train':
            self.csv_file = args.train_csv_file
        elif mode == 'valid':
            self.csv_file = args.valid_csv_file
        elif mode == 'test':
            self.csv_file = args.test_csv_file

        df = pd.read_csv(self.csv_file)
        data = []
        for col in df.columns:
            data.append(df[col].tolist())
        
        #* 制作负样本
        self._make_negative(data)
        

        #* 2、视频特征目录
        self.video_feats_root = args.video_feats_root

        #* 3、加载声韵母列表
        self.phoneme_lst = ["<pad>", ""] + list_of_phoneme

        #* 4、制作 关键词检测和定位的标签
        self._get_labels()
        

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        #* video 特征
        video_feats = np.load("{}/{}.npy".format(self.video_feats_root, self.file_path[index]))
        video_feats = torch.FloatTensor(video_feats)
        
        #* text 特征
        text_feats = [self.phoneme_lst.index(phoneme) for phoneme in pinyins_to_phonemes(str(self.pinyin[index]))]      #! 将self.pinyin[index]转换成str：nan 会被识别为float
        text_feats = torch.tensor(text_feats)

        return video_feats, text_feats, self.detection_label[index], self.location_label[index], self.segment_label[index]
    
    def _get_labels(self):
        """
        制作标签
        """
        self.detection_label = []       #* 关键词检测的标签
        self.location_label = []        #* 关键词定位的标签
        self.segment_label = []         #* 视频字符分割的标签

        for i in range(len(self.file_path)):
            #* 如果是 negative sample
            if self.start[i] == -1 and self.end[i] == -1:
                #* 关键词定位 
                loc_label = torch.zeros(self.duration[i], dtype=torch.bool)
                self.location_label.append(loc_label)
                #* 关键词检测 
                self.detection_label.append(torch.zeros(1, dtype=torch.bool))
                
            #* 如果是 positive sample
            else:
                #* 关键词定位 
                loc_label = torch.zeros(self.duration[i], dtype=torch.bool)
                if self.end[i] < self.duration[i]:
                    loc_label[self.start[i]: self.end[i]+1] = torch.tensor([True] * (self.end[i] - self.start[i] + 1)).bool()
                else:       #! 最后一个关键字的下边界可能是 video 的最后一帧，边界是从0开开，所以应该是最后一帧-1
                    loc_label[self.start[i]: self.duration[i]] = torch.tensor([True] * (self.duration[i] - self.start[i])).bool()
                self.location_label.append(loc_label)
                #* 关键词检测 
                self.detection_label.append(torch.ones(1, dtype=torch.bool))
                
            #* 视频字符分割
            seg_label = torch.zeros(self.duration[i], dtype=torch.bool)

            idx = [int(j) for j in self.segment[i].strip().split('_')]
            for j in idx:
                if j < self.duration[i]:
                    seg_label[j] = True
                else:       #! 最后一个关键字的下边界可能是 video 的最后一帧，边界是从0开开，所以应该是最后一帧-1
                    seg_label[self.duration[i]-1] = True
            self.segment_label.append(seg_label)

    def _make_negative(self, data):
        """
        制作负样本
        """
        file_path, hanzi, pinyin, duration, start, end, positive_words, segment = data

        self.file_path, self.pinyin, self.duration, self.start, self.end, self.segment = [], [], [], [], [], []

        for i in range(len(file_path)):

            #* 1、正常添加正样本
            self.file_path.append(file_path[i])
            self.pinyin.append(pinyin[i])
            self.duration.append(duration[i])
            self.start.append(start[i])
            self.end.append(end[i])
            self.segment.append(segment[i])

            if i == len(file_path)-1: continue

            #* 2、采集同等数量的负样本
            if file_path[i] != file_path[i+1]:
                
                positive_lst = positive_words[i].strip().split("_")
                negative_lst = list(set(list_of_pinyin) - set(positive_lst))
                negative_words = random.sample(negative_lst, len(positive_lst))

                for nw in negative_words:
                    self.file_path.append(file_path[i])
                    self.pinyin.append(nw)
                    self.duration.append(duration[i])
                    self.start.append(-1)
                    self.end.append(-1)
                    self.segment.append(segment[i])