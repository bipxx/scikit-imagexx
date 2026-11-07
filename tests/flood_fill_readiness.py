"""Source-frozen recipe bridge; no scientific import at module entry."""
from flood_fill_contract import recipes, bounds


def cases():
    return recipes()


def source_safe(row):
    assert row in recipes()
    return bounds(row)
