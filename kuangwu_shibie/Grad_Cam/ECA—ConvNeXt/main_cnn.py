import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
from ConvNeXt_model import convnext_tiny
from utils import GradCAM, show_cam_on_image


def main():
    model=convnext_tiny(num_classes=26)
    weights_path = "E:\\李雷\\矿石训练最终程序\\矿石图像训练\\ECA_ConvNeXt\\weights\\best_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    target_layers = [model.attentions[-1]]
    data_transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "ECA—ConvNeXt/0007.JPG"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 3 # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
