# from pypinyin import lazy_pinyin, pinyin
# import pypinyin
# # 将汉字转换成不带声调的拼音
# pinyin = pinyin('战略', style=pypinyin.NORMAL)
# print(pinyin)  # ['zhōng', 'wén']


# with open("dicts/mandarin_pinyin.dict", 'r') as file:
#     data = file.readlines()

# py = []
# for pinyin in data:
#     py.append(pinyin.strip().split(" ")[0])

# py = list(set(py))
# py.sort()

# py_with_tone = ['"' + p + '",' for p in py]
# print(len(py_with_tone))
# print(py_with_tone[:10])
# with open("demo.txt", 'w') as file:
#     file.writelines(py_with_tone)

import timm

print(timm.list_models("swin*")) 

#* 加载model
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
