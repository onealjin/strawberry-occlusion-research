from strawberry_occlusion.geometry import Point, distance, midpoint


def test_distance_uses_euclidean_geometry() -> None:
    start = Point(x=0.0, y=0.0)
    end = Point(x=3.0, y=4.0)

    assert distance(start, end) == 5.0


def test_midpoint_averages_coordinates() -> None:
    start = Point(x=2.0, y=4.0)
    end = Point(x=6.0, y=10.0)

    assert midpoint(start, end) == Point(x=4.0, y=7.0)
