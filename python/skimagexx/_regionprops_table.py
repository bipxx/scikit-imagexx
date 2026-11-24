"""Native label and bounding-box columns for the default region table."""

from . import _core
from ._compat import fallback

_DEFAULT_PROPERTIES = ('label', 'bbox')


def regionprops_table(
    label_image, intensity_image=None, properties=_DEFAULT_PROPERTIES, *,
    cache=True, separator='-', extra_properties=None, spacing=None,
):
    """Measure each positive label and return the table as a dict of columns."""
    default_properties = properties is _DEFAULT_PROPERTIES or (
        type(properties) is tuple and len(properties) == 2
        and type(properties[0]) is str and type(properties[1]) is str
        and properties == _DEFAULT_PROPERTIES
    )
    if (intensity_image is None and default_properties and cache is True
            and type(separator) is str and separator == '-'
            and extra_properties is None and spacing is None):
        result = _core.regionprops_table_default(label_image)
        if result is not None:
            return result
    return fallback(
        'measure', 'regionprops_table', label_image, intensity_image, properties,
        cache=cache, separator=separator, extra_properties=extra_properties, spacing=spacing,
    )
