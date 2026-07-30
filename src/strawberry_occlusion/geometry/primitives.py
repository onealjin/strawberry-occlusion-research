"""Small geometry utilities shared by cutline estimation code."""

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Point:
    """Two-dimensional point in image coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class LineSegment:
    """Finite line segment between two image-coordinate points."""

    start: Point
    end: Point


def distance(start: Point, end: Point) -> float:
    """Return the Euclidean distance between two points."""

    return hypot(end.x - start.x, end.y - start.y)


def midpoint(start: Point, end: Point) -> Point:
    """Return the midpoint between two points."""

    return Point(x=(start.x + end.x) / 2.0, y=(start.y + end.y) / 2.0)
