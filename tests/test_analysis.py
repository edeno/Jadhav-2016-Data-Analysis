from pytest import mark
from src.analysis import is_overlap


@mark.parametrize('interval1, interval2, expected', [
    ((1, 3), (2, 4), True),
    ((1, 3), (4, 5), False),
    ((1, 3), (0, 2), True),
    ((1, 3), (-2, -1), False),
    ((1, 4), (2, 4), True),
])
def test_is_overlap(interval1, interval2, expected):
    assert is_overlap(interval1, interval2) == expected
