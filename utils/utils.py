from pypinyin.style._utils import get_initials, get_finals


def pinyins_to_phonemes(pinyin_str: str):
    """
    将一个拼音序列转换成声韵母序列
    "ni hao wo shi ren" -> ['n', 'i', 'h', 'ao', 'w', 'o', 'sh', 'i', 'r', 'en']
    """
    pinyin_lst = pinyin_str.strip().split(" ")
    phoneme_lst = []
    for pinyin in pinyin_lst:
        phoneme_lst.append(get_initials(pinyin, strict=False))
        phoneme_lst.append(get_finals(pinyin, strict=False))

    return phoneme_lst


def is_valid_pinyin(pinyin_str):
    pinyin_lst = pinyin_str.strip().split(" ")
    flag = True
    for pinyin in pinyin_lst:
        if "".join(pinyins_to_phonemes(pinyin)) != pinyin:
            flag = False
    return flag

# print(pinyins_to_phonemes("nve"))