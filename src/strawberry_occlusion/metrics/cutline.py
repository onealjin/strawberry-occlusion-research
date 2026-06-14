"""Cutline geometry metrics."""

from strawberry_occlusion.geometry import Point, distance


def endpoint_error(
    predicted_start: Point,
    predicted_end: Point,
    target_start: Point,
    target_end: Point,
) -> float:
    """Return the mean endpoint distance between predicted and target cutlines."""

    start_error = distance(predicted_start, target_start)
    end_error = distance(predicted_end, target_end)
    return (start_error + end_error) / 2.0
