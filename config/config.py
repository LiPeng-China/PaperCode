import argparse


def load_arg():
    parse = argparse.ArgumentParser()

    # model 参数
    parse.add_argument("--d_model", type=int, default=512, help=" ")
    parse.add_argument("--nhead", type=int, default=8, help=" ")
    parse.add_argument("--dropout", type=int, default=0.1, help=" ")
    parse.add_argument("--num_layers", type=int, default=6, help=" ")
    parse.add_argument("--vocab", type=int, default=59, help=" ")
    parse.add_argument("--batch_first", type=bool, default=True, help="")

    # dataset
    parse.add_argument("--video_feats_root", type=str, default="/mnt/d/feats", help="")
    parse.add_argument("--train_csv_file", type=str, default="../label/one_word/csv/cmlr_train_one_word.csv", help="")
    parse.add_argument("--valid_csv_file", type=str, default="../label/one_word/csv/cmlr_valid_one_word.csv", help="")
    parse.add_argument("--test_csv_file", type=str, default="../label/one_word/csv/cmlr_test_one_word.csv", help="")

    #* dataloader
    parse.add_argument("--batch_size", type=int, default=8, help="")
    

    args = parse.parse_args()
    return args
