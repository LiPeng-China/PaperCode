import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


from dataset import MyDataset
import sys
sys.path.append(r"/mnt/c/Users/peng/Code/PaperCode/config")
from config import load_arg



# valid_set = MyDataset(args, mode='valid')

def my_collate(batch):
    video_feats = [b[0] for b in batch]             #* shape = (b, T, 512)
    text_feats = [b[1] for b in batch]              #* shape = (b, 2) 2 指的是声母和韵母的index
    detection_label = [b[2] for b in batch]         #* shape = (b, 1) 1:True或False
    location_label = [b[3] for b in batch]          #* shape = (b, T) T: duration 持续时间 True或False
    segment_label = [b[4] for b in batch]           #* shape = (b, T) T: duration 持续时间 True或False

    max_duration = max(len(duration) for duration in video_feats)

    #! src_key_padding_mask 是一个二值化的tensor，在需要被忽略地方应该是True，在需要保留原值的情况下，是False
    video_src = pad_sequence(video_feats, batch_first=True, padding_value=torch.inf)
    video_src_key_padding_mask = (video_src==torch.inf)         # mask 的内容到底哪里是true， 哪里是false
    
    text_src = pad_sequence(text_feats, batch_first=True, padding_value=0)
    text_src_key_padding_mask = (text_src == 0)                 # mask 的内容到底哪里是true， 哪里是false
    
    

    return video_src, video_src_key_padding_mask, text_src, text_src_key_padding_mask, detection_label, location_label, segment_label


# valid_loader = DataLoader(valid_set, batch_size=2, shuffle=True, collate_fn=my_collate)


def init_loader(args, mode):
    data_set = MyDataset(args, mode=mode)
    data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)
    return data_loader


args = load_arg()
for video_src, video_mask, text_src, text_mask, detection_label, location_label, segment_label in init_loader(args, "valid"):
    print(type(video_src))
    print(video_src.shape)
    print(type(video_mask))
    print(video_mask.shape)
    print(type(text_src))
    print(text_src.shape)
    print(type(text_mask))
    print(text_mask.shape)
    break
    
