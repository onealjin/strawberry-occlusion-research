"""Deterministic cutline geometry derived from visible semantic masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite
from numbers import Real
from typing import Literal

import numpy as np

from strawberry_occlusion.geometry.primitives import LineSegment, Point


CLASS_MAPPING = {"background": 0, "Flesh": 1, "Calyx": 2}
_DIRECTION_EPSILON = 1e-12


@dataclass(frozen=True)
class MaskCutlineParameters:
    """Parameters controlling visible-mask cutline estimation."""

    calyx_dilation_radius: int = 1
    signed_offset: float = 0.0
    component_connectivity: Literal[4, 8] = 8


@dataclass(frozen=True)
class MaskCutlineResult:
    """Geometry, diagnostics, and status for one visible semantic mask.

    Baseline v1 records disconnected regions in the selected contact band but
    deliberately merges them when calculating one combined attachment centroid.
    """

    status: Literal["ok", "failed"]
    failure_code: str | None
    failure_reason: str | None
    parameters: MaskCutlineParameters
    image_width: int
    image_height: int
    flesh_pixel_count: int
    calyx_pixel_count: int
    flesh_component_count: int
    calyx_component_count: int
    selected_flesh_component: int | None
    selected_calyx_component: int | None
    selected_flesh_pixel_count: int
    selected_calyx_pixel_count: int
    contact_pixel_count: int
    contact_component_count: int
    flesh_centroid: Point | None
    attachment_anchor: Point | None
    fruit_to_attachment_direction: tuple[float, float] | None
    candidate_cutline: LineSegment | None
    offset_anchor: Point | None
    final_cutline: LineSegment | None
    flesh_mask: np.ndarray = field(repr=False, compare=False)
    calyx_mask: np.ndarray = field(repr=False, compare=False)
    contact_band: np.ndarray = field(repr=False, compare=False)

    @property
    def succeeded(self) -> bool:
        """Return whether a final clipped cutline was produced."""

        return self.status == "ok"


@dataclass(frozen=True)
class _Component:
    label: int
    size: int
    pixels: np.ndarray = field(repr=False, compare=False)


def estimate_visible_mask_cutline(
    mask: np.ndarray,
    *,
    calyx_dilation_radius: int = 1,
    signed_offset: float = 0.0,
    component_connectivity: Literal[4, 8] = 8,
) -> MaskCutlineResult:
    """Estimate a perpendicular cutline from visible Flesh and Calyx classes.

    Positive offsets move the final cutline along the normalized
    fruit-to-attachment direction. Negative offsets move it toward the selected
    Flesh component's centroid. Coordinates use pixel centres, so image bounds
    are ``x in [0, width - 1]`` and ``y in [0, height - 1]``. The Calyx
    dilation radius must be at least one pixel because mutually exclusive
    semantic classes cannot overlap before dilation.
    """

    mask_array = _validate_mask(mask)
    parameters = _validate_parameters(
        calyx_dilation_radius=calyx_dilation_radius,
        signed_offset=signed_offset,
        component_connectivity=component_connectivity,
    )
    height, width = mask_array.shape
    flesh_mask = mask_array == CLASS_MAPPING["Flesh"]
    calyx_mask = mask_array == CLASS_MAPPING["Calyx"]
    flesh_labels, flesh_components = _label_components(
        flesh_mask,
        connectivity=parameters.component_connectivity,
    )
    _, calyx_components = _label_components(
        calyx_mask,
        connectivity=parameters.component_connectivity,
    )
    empty_contact = np.zeros(mask_array.shape, dtype=bool)
    common = {
        "parameters": parameters,
        "image_width": width,
        "image_height": height,
        "flesh_pixel_count": int(np.count_nonzero(flesh_mask)),
        "calyx_pixel_count": int(np.count_nonzero(calyx_mask)),
        "flesh_component_count": len(flesh_components),
        "calyx_component_count": len(calyx_components),
        "flesh_mask": _readonly_boolean_copy(flesh_mask),
        "calyx_mask": _readonly_boolean_copy(calyx_mask),
    }

    if not flesh_components:
        return _result(
            **common,
            status="failed",
            failure_code="no_flesh",
            failure_reason="Mask contains no Flesh pixels (class 1).",
            contact_band=empty_contact,
        )
    if not calyx_components:
        return _result(
            **common,
            status="failed",
            failure_code="no_calyx",
            failure_reason="Mask contains no Calyx pixels (class 2).",
            contact_band=empty_contact,
        )

    selected_pair = _select_contacting_components(
        flesh_labels,
        flesh_components,
        calyx_components,
        dilation_radius=parameters.calyx_dilation_radius,
    )
    if selected_pair is None:
        return _result(
            **common,
            status="failed",
            failure_code="no_contact",
            failure_reason=(
                "No Flesh/Calyx contact was found with Calyx dilation radius "
                f"{parameters.calyx_dilation_radius}."
            ),
            contact_band=empty_contact,
        )

    selected_flesh, selected_calyx, contact_band = selected_pair
    selected_flesh_mask = flesh_labels == selected_flesh.label
    flesh_centroid = _mask_centroid(selected_flesh_mask)
    attachment_anchor = _mask_centroid(contact_band)
    delta_x = attachment_anchor.x - flesh_centroid.x
    delta_y = attachment_anchor.y - flesh_centroid.y
    direction_length = hypot(delta_x, delta_y)
    selected = {
        "selected_flesh": selected_flesh,
        "selected_calyx": selected_calyx,
        "contact_band": contact_band,
        "flesh_centroid": flesh_centroid,
        "attachment_anchor": attachment_anchor,
    }
    if direction_length <= _DIRECTION_EPSILON:
        return _result(
            **common,
            **selected,
            status="failed",
            failure_code="degenerate_direction",
            failure_reason=(
                "Flesh centroid and attachment anchor coincide; "
                "fruit-to-attachment direction is undefined."
            ),
        )

    direction = (delta_x / direction_length, delta_y / direction_length)
    perpendicular = (-direction[1], direction[0])
    candidate_cutline = clip_infinite_line_to_image(
        attachment_anchor,
        perpendicular,
        width=width,
        height=height,
    )
    if candidate_cutline is None:
        return _result(
            **common,
            **selected,
            status="failed",
            failure_code="candidate_line_outside_image",
            failure_reason="Candidate cutline does not cross the image bounds.",
            fruit_to_attachment_direction=direction,
        )

    offset_anchor = Point(
        x=attachment_anchor.x + parameters.signed_offset * direction[0],
        y=attachment_anchor.y + parameters.signed_offset * direction[1],
    )
    final_cutline = clip_infinite_line_to_image(
        offset_anchor,
        perpendicular,
        width=width,
        height=height,
    )
    if final_cutline is None:
        return _result(
            **common,
            **selected,
            status="failed",
            failure_code="offset_line_outside_image",
            failure_reason=(
                "Final offset cutline does not cross the image bounds; "
                f"signed offset was {parameters.signed_offset}."
            ),
            fruit_to_attachment_direction=direction,
            candidate_cutline=candidate_cutline,
            offset_anchor=offset_anchor,
        )

    return _result(
        **common,
        **selected,
        status="ok",
        failure_code=None,
        failure_reason=None,
        fruit_to_attachment_direction=direction,
        candidate_cutline=candidate_cutline,
        offset_anchor=offset_anchor,
        final_cutline=final_cutline,
    )


def clip_infinite_line_to_image(
    point: Point,
    direction: tuple[float, float],
    *,
    width: int,
    height: int,
) -> LineSegment | None:
    """Clip an infinite parametric line to pixel-centre image bounds.

    Endpoints are ordered by increasing parameter ``t`` in
    ``point + t * direction``. Reversing ``direction`` therefore reverses the
    returned endpoints. Future cutline-comparison metrics should be invariant to
    endpoint order.
    """

    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width < 2
        or height < 2
    ):
        raise ValueError("width and height must be integers of at least 2")
    if len(direction) != 2 or not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and isfinite(float(value))
        for value in direction
    ):
        raise ValueError("direction must contain two finite real values")
    direction_x, direction_y = (float(direction[0]), float(direction[1]))
    if hypot(direction_x, direction_y) <= _DIRECTION_EPSILON:
        raise ValueError("direction must be non-zero")

    max_x = float(width - 1)
    max_y = float(height - 1)
    intersections: list[tuple[float, Point]] = []
    if abs(direction_x) > _DIRECTION_EPSILON:
        for x in (0.0, max_x):
            parameter = (x - point.x) / direction_x
            y = point.y + parameter * direction_y
            if -_DIRECTION_EPSILON <= y <= max_y + _DIRECTION_EPSILON:
                intersections.append((parameter, Point(x=x, y=min(max(y, 0.0), max_y))))
    if abs(direction_y) > _DIRECTION_EPSILON:
        for y in (0.0, max_y):
            parameter = (y - point.y) / direction_y
            x = point.x + parameter * direction_x
            if -_DIRECTION_EPSILON <= x <= max_x + _DIRECTION_EPSILON:
                intersections.append((parameter, Point(x=min(max(x, 0.0), max_x), y=y)))

    unique: list[tuple[float, Point]] = []
    for parameter, intersection in sorted(intersections, key=lambda item: item[0]):
        if (
            not unique
            or hypot(
                intersection.x - unique[-1][1].x,
                intersection.y - unique[-1][1].y,
            )
            > _DIRECTION_EPSILON
        ):
            unique.append((parameter, intersection))
    if len(unique) < 2:
        return None
    return LineSegment(start=unique[0][1], end=unique[-1][1])


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy.ndarray, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [H, W], got {mask.shape}")
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        raise ValueError(
            f"mask height and width must both be at least 2, got {mask.shape}"
        )
    if np.issubdtype(mask.dtype, np.bool_) or not np.issubdtype(mask.dtype, np.integer):
        raise TypeError(f"mask must use an integer dtype, got {mask.dtype}")
    valid = np.isin(mask, tuple(CLASS_MAPPING.values()))
    if not np.all(valid):
        invalid_values = np.unique(mask[~valid]).tolist()
        raise ValueError(
            f"mask must contain only class IDs 0, 1, and 2, found {invalid_values}"
        )
    return mask


def _validate_parameters(
    *,
    calyx_dilation_radius: int,
    signed_offset: float,
    component_connectivity: int,
) -> MaskCutlineParameters:
    if (
        isinstance(calyx_dilation_radius, bool)
        or not isinstance(calyx_dilation_radius, int)
        or calyx_dilation_radius < 1
    ):
        raise ValueError("calyx_dilation_radius must be an integer of at least 1")
    if (
        isinstance(signed_offset, bool)
        or not isinstance(signed_offset, Real)
        or not isfinite(float(signed_offset))
    ):
        raise ValueError("signed_offset must be a finite real number")
    if (
        isinstance(component_connectivity, bool)
        or not isinstance(component_connectivity, int)
        or component_connectivity not in (4, 8)
    ):
        raise ValueError("component_connectivity must be 4 or 8")
    return MaskCutlineParameters(
        calyx_dilation_radius=calyx_dilation_radius,
        signed_offset=float(signed_offset),
        component_connectivity=component_connectivity,
    )


def _label_components(
    binary_mask: np.ndarray,
    *,
    connectivity: Literal[4, 8],
) -> tuple[np.ndarray, tuple[_Component, ...]]:
    labels = np.zeros(binary_mask.shape, dtype=np.int32)
    components: list[_Component] = []
    neighbours = (
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        if connectivity == 4
        else (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    )
    height, width = binary_mask.shape
    for start_y, start_x in np.argwhere(binary_mask):
        y = int(start_y)
        x = int(start_x)
        if labels[y, x] != 0:
            continue
        label = len(components) + 1
        labels[y, x] = label
        queue: deque[tuple[int, int]] = deque([(y, x)])
        pixels: list[tuple[int, int]] = []
        while queue:
            pixel_y, pixel_x = queue.popleft()
            pixels.append((pixel_y, pixel_x))
            for delta_y, delta_x in neighbours:
                neighbour_y = pixel_y + delta_y
                neighbour_x = pixel_x + delta_x
                if (
                    0 <= neighbour_y < height
                    and 0 <= neighbour_x < width
                    and binary_mask[neighbour_y, neighbour_x]
                    and labels[neighbour_y, neighbour_x] == 0
                ):
                    labels[neighbour_y, neighbour_x] = label
                    queue.append((neighbour_y, neighbour_x))
        pixel_array = np.asarray(pixels, dtype=np.int32)
        components.append(
            _Component(
                label=label,
                size=len(pixels),
                pixels=pixel_array,
            )
        )
    return labels, tuple(components)


def _select_contacting_components(
    flesh_labels: np.ndarray,
    flesh_components: tuple[_Component, ...],
    calyx_components: tuple[_Component, ...],
    *,
    dilation_radius: int,
) -> tuple[_Component, _Component, np.ndarray] | None:
    shape = flesh_labels.shape
    flesh_by_label = {component.label: component for component in flesh_components}
    best_key: tuple[int, int, int, int, int] | None = None
    best_pair: tuple[_Component, _Component, np.ndarray] | None = None
    for calyx_component in calyx_components:
        component_mask = np.zeros(shape, dtype=bool)
        component_mask[
            calyx_component.pixels[:, 0],
            calyx_component.pixels[:, 1],
        ] = True
        dilated_calyx = _binary_square_dilation(
            component_mask,
            radius=dilation_radius,
        )
        contacted_labels = flesh_labels[dilated_calyx & (flesh_labels > 0)]
        counts = np.bincount(
            contacted_labels,
            minlength=len(flesh_components) + 1,
        )
        for flesh_label in np.flatnonzero(counts[1:]) + 1:
            flesh_component = flesh_by_label[int(flesh_label)]
            contact_band = dilated_calyx & (flesh_labels == flesh_label)
            key = (
                int(counts[flesh_label]),
                flesh_component.size,
                calyx_component.size,
                -flesh_component.label,
                -calyx_component.label,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (
                    flesh_component,
                    calyx_component,
                    contact_band,
                )
    return best_pair


def _binary_square_dilation(binary_mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius == 0:
        return binary_mask.copy()
    height, width = binary_mask.shape
    padded = np.pad(binary_mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(binary_mask)
    diameter = 2 * radius + 1
    for offset_y in range(diameter):
        for offset_x in range(diameter):
            dilated |= padded[
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
    return dilated


def _mask_centroid(binary_mask: np.ndarray) -> Point:
    coordinates = np.argwhere(binary_mask)
    return Point(
        x=float(coordinates[:, 1].mean()),
        y=float(coordinates[:, 0].mean()),
    )


def _readonly_boolean_copy(mask: np.ndarray) -> np.ndarray:
    copied = np.array(mask, dtype=bool, copy=True)
    copied.setflags(write=False)
    return copied


def _result(
    *,
    status: Literal["ok", "failed"],
    failure_code: str | None,
    failure_reason: str | None,
    parameters: MaskCutlineParameters,
    image_width: int,
    image_height: int,
    flesh_pixel_count: int,
    calyx_pixel_count: int,
    flesh_component_count: int,
    calyx_component_count: int,
    flesh_mask: np.ndarray,
    calyx_mask: np.ndarray,
    contact_band: np.ndarray,
    selected_flesh: _Component | None = None,
    selected_calyx: _Component | None = None,
    flesh_centroid: Point | None = None,
    attachment_anchor: Point | None = None,
    fruit_to_attachment_direction: tuple[float, float] | None = None,
    candidate_cutline: LineSegment | None = None,
    offset_anchor: Point | None = None,
    final_cutline: LineSegment | None = None,
) -> MaskCutlineResult:
    _, contact_components = _label_components(
        contact_band,
        connectivity=parameters.component_connectivity,
    )
    return MaskCutlineResult(
        status=status,
        failure_code=failure_code,
        failure_reason=failure_reason,
        parameters=parameters,
        image_width=image_width,
        image_height=image_height,
        flesh_pixel_count=flesh_pixel_count,
        calyx_pixel_count=calyx_pixel_count,
        flesh_component_count=flesh_component_count,
        calyx_component_count=calyx_component_count,
        selected_flesh_component=(
            selected_flesh.label if selected_flesh is not None else None
        ),
        selected_calyx_component=(
            selected_calyx.label if selected_calyx is not None else None
        ),
        selected_flesh_pixel_count=(
            selected_flesh.size if selected_flesh is not None else 0
        ),
        selected_calyx_pixel_count=(
            selected_calyx.size if selected_calyx is not None else 0
        ),
        contact_pixel_count=int(np.count_nonzero(contact_band)),
        contact_component_count=len(contact_components),
        flesh_centroid=flesh_centroid,
        attachment_anchor=attachment_anchor,
        fruit_to_attachment_direction=fruit_to_attachment_direction,
        candidate_cutline=candidate_cutline,
        offset_anchor=offset_anchor,
        final_cutline=final_cutline,
        flesh_mask=flesh_mask,
        calyx_mask=calyx_mask,
        contact_band=_readonly_boolean_copy(contact_band),
    )
