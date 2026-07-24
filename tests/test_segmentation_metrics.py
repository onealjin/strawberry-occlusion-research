import pytest
import torch

from strawberry_occlusion.metrics import (
    confusion_matrix,
    dice_score,
    intersection_over_union,
    mean_ignore_nan,
)


def test_perfect_prediction() -> None:
    target = torch.tensor([[0, 1], [1, 2]])
    predicted = target.clone()

    matrix = confusion_matrix(predicted, target, num_classes=3)

    assert torch.equal(matrix, torch.diag(torch.tensor([1, 2, 1])))
    assert matrix.dtype == torch.long
    torch.testing.assert_close(
        intersection_over_union(matrix), torch.ones(3, dtype=torch.float64)
    )
    torch.testing.assert_close(dice_score(matrix), torch.ones(3, dtype=torch.float64))


def test_completely_incorrect_prediction() -> None:
    target = torch.tensor([[0, 0], [1, 1]])
    predicted = torch.tensor([[1, 1], [0, 0]])

    matrix = confusion_matrix(predicted, target, num_classes=2)

    assert torch.equal(matrix, torch.tensor([[0, 2], [2, 0]]))
    torch.testing.assert_close(
        intersection_over_union(matrix), torch.zeros(2, dtype=torch.float64)
    )
    torch.testing.assert_close(dice_score(matrix), torch.zeros(2, dtype=torch.float64))


def test_mixed_multiclass_prediction() -> None:
    target = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    predicted = torch.tensor([[0, 1, 2], [0, 1, 1], [0, 2, 2]])

    matrix = confusion_matrix(predicted, target, num_classes=3)

    assert torch.equal(
        matrix,
        torch.tensor(
            [
                [1, 1, 1],
                [1, 2, 0],
                [1, 0, 2],
            ]
        ),
    )
    torch.testing.assert_close(
        intersection_over_union(matrix),
        torch.tensor([1 / 5, 1 / 2, 1 / 2], dtype=torch.float64),
    )
    torch.testing.assert_close(
        dice_score(matrix),
        torch.tensor([1 / 3, 2 / 3, 2 / 3], dtype=torch.float64),
    )


def test_batched_masks() -> None:
    target = torch.tensor([[[0, 1]], [[1, 0]]])
    predicted = torch.tensor([[[0, 0]], [[1, 1]]])

    matrix = confusion_matrix(predicted, target, num_classes=2)

    assert torch.equal(matrix, torch.tensor([[1, 1], [1, 1]]))
    torch.testing.assert_close(
        intersection_over_union(matrix),
        torch.tensor([1 / 3, 1 / 3], dtype=torch.float64),
    )
    torch.testing.assert_close(
        dice_score(matrix), torch.tensor([1 / 2, 1 / 2], dtype=torch.float64)
    )


def test_ignore_index_excludes_target_pixels() -> None:
    target = torch.tensor([[0, 255], [1, 255]])
    predicted = torch.tensor([[0, 99], [0, -5]])

    matrix = confusion_matrix(predicted, target, num_classes=2, ignore_index=255)

    assert torch.equal(matrix, torch.tensor([[1, 0], [1, 0]]))


def test_absent_classes_produce_nan() -> None:
    target = torch.tensor([[0, 0]])
    predicted = target.clone()
    matrix = confusion_matrix(predicted, target, num_classes=3)

    iou = intersection_over_union(matrix)
    dice = dice_score(matrix)

    assert iou[0] == 1
    assert dice[0] == 1
    assert torch.isnan(iou[1:]).all()
    assert torch.isnan(dice[1:]).all()
    assert mean_ignore_nan(iou) == 1
    assert mean_ignore_nan(dice) == 1


def test_prediction_only_class_is_zero_while_absent_class_is_nan() -> None:
    target = torch.tensor([[0, 0]])
    predicted = torch.tensor([[0, 1]])
    matrix = confusion_matrix(predicted, target, num_classes=3)

    iou = intersection_over_union(matrix)
    dice = dice_score(matrix)

    assert torch.equal(
        matrix,
        torch.tensor(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert iou.dtype == torch.float64
    assert dice.dtype == torch.float64
    assert iou[1] == 0
    assert dice[1] == 0
    assert torch.isnan(iou[2])
    assert torch.isnan(dice[2])


def test_mean_ignore_nan_uses_only_present_values() -> None:
    values = torch.tensor([0.5, torch.nan, 1.0])

    assert mean_ignore_nan(values) == pytest.approx(0.75)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        confusion_matrix(
            torch.zeros(2, 2, dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.long),
            num_classes=2,
        )


@pytest.mark.parametrize(
    ("predicted", "target", "message"),
    [
        (
            torch.zeros(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            "predicted must use an integer dtype",
        ),
        (
            torch.zeros(2, 2, dtype=torch.long),
            torch.zeros(2, 2),
            "target must use an integer dtype",
        ),
    ],
)
def test_invalid_mask_dtypes_are_rejected(
    predicted: torch.Tensor,
    target: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        confusion_matrix(predicted, target, num_classes=2)


@pytest.mark.parametrize(
    ("predicted", "target", "message"),
    [
        (
            torch.tensor([[0, 2]]),
            torch.tensor([[0, 1]]),
            "predicted labels",
        ),
        (
            torch.tensor([[0, 1]]),
            torch.tensor([[0, -1]]),
            "target labels",
        ),
    ],
)
def test_out_of_range_labels_are_rejected(
    predicted: torch.Tensor,
    target: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        confusion_matrix(predicted, target, num_classes=2)


@pytest.mark.parametrize("num_classes", [0, -1])
def test_invalid_num_classes_values_are_rejected(num_classes: int) -> None:
    masks = torch.zeros(2, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="at least 1"):
        confusion_matrix(masks, masks, num_classes=num_classes)


def test_non_integer_num_classes_is_rejected() -> None:
    masks = torch.zeros(2, 2, dtype=torch.long)

    with pytest.raises(TypeError, match="must be an integer"):
        confusion_matrix(masks, masks, num_classes=2.0)  # type: ignore[arg-type]


def test_invalid_mask_rank_is_rejected() -> None:
    masks = torch.zeros(1, 2, 2, 2, dtype=torch.long)

    with pytest.raises(ValueError, match=r"\[H, W\].*\[B, H, W\]"):
        confusion_matrix(masks, masks, num_classes=2)
