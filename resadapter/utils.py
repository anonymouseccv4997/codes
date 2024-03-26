from PIL import ImageDraw, ImageFont

from torchvision import transforms

def draw_text_on_images(image, text):
    pil_image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(pil_image)
    max_length = max(image.shape[-2], image.shape[-1])

    if max_length >= 512:
        font_scale = 0.08
    else:
        font_scale = 0.1
    font_size = int(font_scale * max_length)
    font = ImageFont.truetype("assets/Chalkduster.ttf", size=font_size)

    offset = 10
    x = offset
    y = offset

    draw.text((x, y), text, fill=(56, 136, 239), font=font)
    image = transforms.ToTensor()(pil_image)
    return image