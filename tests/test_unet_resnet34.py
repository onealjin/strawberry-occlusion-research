import torch

from strawberry_occlusion.models import ResNet34UNet


def test_resnet34_unet_outputs_input_spatial_shape() -> None:
    model = ResNet34UNet(num_classes=4)
    model.eval()
    image = torch.randn(2, 3, 96, 128)

    with torch.no_grad():
        logits = model(image)

    assert logits.shape == (2, 4, 96, 128)
