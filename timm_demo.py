import timm
import torch
from torchvision import transforms as T
from PIL import Image

#* 读取图片
image = Image.open('./11.jpg')
#* 160*80, JPEG, RGB, 
# print(type(image), image.size, image.format, image.mode, image.getbands())
#* 输入预处理

trans_func = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)])

transed_image = trans_func(image)

batch_input = transed_image.unsqueeze(0)
print(batch_input.shape)

#* 模型
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

#* 推理及预测

with torch.no_grad():
    output = model(batch_input)

print(output.shape)
