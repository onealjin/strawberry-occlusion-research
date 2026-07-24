import pytest
import torch
from torch import nn

from strawberry_occlusion.inference import predict_segmentation_masks


class ToySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("class_offsets", torch.tensor([0.0, 1.0, -1.0]))
        self.input_shape: tuple[int, ...] | None = None
        self.used_inference_mode = False
        self.was_training_during_forward = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(image.shape)
        self.used_inference_mode = torch.is_inference_mode_enabled()
        self.was_training_during_forward = self.training
        return image[:, :1] + self.class_offsets.view(1, 3, 1, 1)


class StaticOutputModel(nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output

    def forward(self, image: torch.Tensor) -> object:
        return self.output


def test_predict_segmentation_masks_accepts_single_image() -> None:
    model = ToySegmentationModel()
    image = torch.zeros(3, 4, 5)

    masks = predict_segmentation_masks(model, image)

    assert masks.shape == (1, 4, 5)
    assert masks.dtype == torch.long
    assert torch.equal(masks, torch.ones(1, 4, 5, dtype=torch.long))
    assert model.input_shape == (1, 3, 4, 5)
    assert model.used_inference_mode
    assert not model.was_training_during_forward
    assert model.training


def test_predict_segmentation_masks_accepts_batch_and_explicit_device() -> None:
    model = ToySegmentationModel()
    model.eval()
    images = torch.zeros(2, 3, 3, 4)

    masks = predict_segmentation_masks(model, images, device=torch.device("cpu"))

    assert masks.shape == (2, 3, 4)
    assert masks.device.type == "cpu"
    assert not model.training


def test_predict_segmentation_masks_returns_batched_argmax_classes() -> None:
    images = torch.tensor(
        [
            [[[3.0, 0.0]], [[1.0, 2.0]], [[2.0, 4.0]]],
            [[[1.0, 5.0]], [[3.0, 2.0]], [[2.0, 1.0]]],
        ]
    )

    masks = predict_segmentation_masks(nn.Identity(), images)

    expected = torch.tensor([[[0, 2]], [[1, 0]]], dtype=torch.long)
    assert torch.equal(masks, expected)


def test_predict_segmentation_masks_accepts_dictionary_output() -> None:
    logits = torch.zeros(1, 3, 4, 5)
    logits[:, 2] = 1.0

    masks = predict_segmentation_masks(
        StaticOutputModel({"logits": logits}), torch.zeros(3, 4, 5)
    )

    assert torch.equal(masks, torch.full((1, 4, 5), 2, dtype=torch.long))


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (torch.zeros(3, 4), "shape"),
        (torch.zeros(1, 4, 5), "exactly 3 channels"),
        (torch.zeros(2, 1, 4, 5), "exactly 3 channels"),
        (torch.zeros(0, 3, 4, 5), "batch dimension"),
        (torch.zeros(3, 0, 5), "height and width"),
        (torch.zeros(3, 4, 0), "height and width"),
    ],
)
def test_predict_segmentation_masks_rejects_invalid_input_shapes(
    image: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predict_segmentation_masks(ToySegmentationModel(), image)


@pytest.mark.parametrize(
    ("output", "error_type", "message"),
    [
        ({}, KeyError, "'logits' key"),
        ({"logits": "not a tensor"}, TypeError, "value for 'logits'"),
        ("not logits", TypeError, "logits torch.Tensor"),
        (torch.zeros(1, 4, 5), ValueError, r"\[B, C, H, W\]"),
        (torch.zeros(2, 3, 4, 5), ValueError, "batch dimension"),
        (torch.zeros(1, 0, 4, 5), ValueError, "class dimension"),
        (torch.zeros(1, 3, 2, 5), ValueError, "height and width"),
    ],
)
def test_predict_segmentation_masks_rejects_invalid_model_outputs(
    output: object,
    error_type: type[Exception],
    message: str,
) -> None:
    model = StaticOutputModel(output)

    with pytest.raises(error_type, match=message):
        predict_segmentation_masks(model, torch.zeros(3, 4, 5))

    assert model.training


def test_predict_segmentation_masks_validates_argument_types() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        predict_segmentation_masks(object(), torch.zeros(3, 4, 5))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="torch.Tensor"):
        predict_segmentation_masks(ToySegmentationModel(), object())  # type: ignore[arg-type]
