"""Pure-Python compatibility implementations of wrf-python interpolation APIs."""

from __future__ import annotations

import numpy as np


# netCDF's default double-precision fill value, used by wrf-python 1.3.4.1.
DEFAULT_FILL_FLOAT64 = 9.969209968386869e36


def _load_xarray():
    """Return xarray when it is installed, otherwise return ``None``."""
    try:
        import xarray as xr
    except ImportError:
        return None
    return xr


def _validate_inputs(field3d, vert):
    field = np.asarray(field3d)
    coordinate = np.asarray(vert)

    if coordinate.ndim < 3:
        raise ValueError(
            "vert must have at least three dimensions with rightmost "
            "dimensions (nz, ny, nx)"
        )
    if field.ndim < 3:
        raise ValueError(
            "field3d must have at least three dimensions with rightmost "
            "dimensions (nz, ny, nx)"
        )

    multiproduct = field.ndim == coordinate.ndim + 1
    if multiproduct:
        if field.shape[1:] != coordinate.shape:
            raise ValueError(
                "field3d with a product dimension must have shape "
                f"(nproduct, *vert.shape); got {field.shape} and "
                f"{coordinate.shape}"
            )
    elif field.shape != coordinate.shape:
        raise ValueError(
            "field3d and vert must have the same shape, except for one "
            f"optional leading product dimension; got {field.shape} and "
            f"{coordinate.shape}"
        )

    nz, ny, nx = coordinate.shape[-3:]
    if nz == 0 or ny == 0 or nx == 0:
        raise ValueError(
            "field3d and vert rightmost dimensions (nz, ny, nx) must be non-empty"
        )

    return field, coordinate, multiproduct


def _normalize_levels(desiredlev, coordinate_shape):
    levels = np.asarray(desiredlev)
    if levels.ndim == 0:
        levels = np.asarray([desiredlev], dtype=np.float64)
        return levels, False

    if levels.ndim == 1:
        return levels.astype(np.float64, copy=False), False

    left_shape = coordinate_shape[:-3]
    horizontal_shape = coordinate_shape[-2:]
    if levels.shape[-2:] != horizontal_shape:
        raise ValueError(
            "desiredlev target-surface rightmost dimensions must match "
            f"(ny, nx)={horizontal_shape}; got {levels.shape[-2:]}"
        )
    if levels.ndim > 2 and levels.shape[:-2] != left_shape:
        raise ValueError(
            "desiredlev target-surface left dimensions must match vert; "
            f"got {levels.shape[:-2]} and {left_shape}"
        )

    return levels.astype(np.float64, copy=False), True


def _interpolate_levels(field, coordinate, levels, missing):
    """Translate pinned DINTERP3DZ for one ``(nz, ny, nx)`` field."""
    nz, ny, nx = coordinate.shape
    output = np.full((levels.size, ny, nx), missing, dtype=np.float64)

    # DINTERP3DZ selects direction from the first horizontal column and scans
    # from model top toward the bottom. Its comparisons are strictly inside a
    # layer; a desired level exactly equal to a model level remains missing.
    descending = coordinate[0, 0, 0] > coordinate[-1, 0, 0]
    for level_index, desired in enumerate(levels):
        found = np.zeros((ny, nx), dtype=bool)
        output_plane = output[level_index]

        for k in range(nz - 1, 0, -1):
            if descending:
                low_z = coordinate[k]
                high_z = coordinate[k - 1]
                low_field = field[k]
                high_field = field[k - 1]
            else:
                low_z = coordinate[k - 1]
                high_z = coordinate[k]
                low_field = field[k - 1]
                high_field = field[k]

            matches = (low_z < desired) & (high_z > desired) & ~found
            if not np.any(matches):
                continue

            high_weight = (desired - low_z[matches]) / (
                high_z[matches] - low_z[matches]
            )
            output_plane[matches] = (
                (1.0 - high_weight) * low_field[matches]
                + high_weight * high_field[matches]
            )
            found |= matches
            if found.all():
                break

    return output


def _interpolate_surface(field, coordinate, target, missing):
    """Translate pinned DINTERP3DZ_2DLEV for one target surface."""
    nz, ny, nx = coordinate.shape
    output = np.full((ny, nx), missing, dtype=np.float64)
    found = np.zeros(output.shape, dtype=bool)

    descending = coordinate[0, 0, 0] > coordinate[-1, 0, 0]
    for k in range(nz - 1, 0, -1):
        if descending:
            low_z = coordinate[k]
            high_z = coordinate[k - 1]
            low_field = field[k]
            high_field = field[k - 1]
        else:
            low_z = coordinate[k - 1]
            high_z = coordinate[k]
            low_field = field[k - 1]
            high_field = field[k]

        matches = (low_z < target) & (high_z > target) & ~found
        if not np.any(matches):
            continue

        high_weight = (target[matches] - low_z[matches]) / (
            high_z[matches] - low_z[matches]
        )
        output[matches] = (
            (1.0 - high_weight) * low_field[matches]
            + high_weight * high_field[matches]
        )
        found |= matches
        if found.all():
            break

    return output


def _interpolate_all(field, coordinate, levels, levels_are_surfaces, missing):
    left_shape = coordinate.shape[:-3]
    horizontal_shape = coordinate.shape[-2:]
    multiproduct = field.ndim == coordinate.ndim + 1
    product_shape = field.shape[:1] if multiproduct else ()

    if levels_are_surfaces:
        output_shape = product_shape + left_shape + horizontal_shape
    else:
        output_shape = product_shape + left_shape + (levels.size,) + horizontal_shape
    # The pinned left-iteration decorator exposes its float64 work array for
    # a product field with no left dimensions. Other paths cast back to the
    # field dtype.
    output_dtype = np.float64 if multiproduct and not left_shape else field.dtype
    output = np.empty(output_shape, dtype=output_dtype)

    for left_index in np.ndindex(left_shape):
        coordinate_slice = np.asarray(coordinate[left_index], dtype=np.float64)
        if levels_are_surfaces:
            target = levels if levels.ndim == 2 else levels[left_index]

        product_indexes = range(field.shape[0]) if multiproduct else (None,)
        for product_index in product_indexes:
            if multiproduct:
                field_index = (product_index,) + left_index
                output_index = field_index
            else:
                field_index = left_index
                output_index = left_index

            field_slice = np.asarray(field[field_index], dtype=np.float64)
            if levels_are_surfaces:
                result = _interpolate_surface(
                    field_slice, coordinate_slice, target, missing
                )
            else:
                result = _interpolate_levels(
                    field_slice, coordinate_slice, levels, missing
                )
            output[output_index] = result

    return output


def _metadata_result(
    result,
    field3d,
    vert,
    desiredlev,
    levels,
    levels_are_surfaces,
    missing,
    squeeze,
    multiproduct,
    xr,
):
    attrs = {}
    dims = None
    coords = None
    name = "field3d_interp"
    requested_levels = np.asarray(desiredlev)

    if isinstance(field3d, xr.DataArray):
        dims = list(field3d.dims)
        vertical_dim = dims[-3]
        del dims[-3]

        coords = dict(field3d.coords)
        coords.pop(vertical_dim, None)

        if not levels_are_surfaces:
            dims.insert(-2, "level")
            if requested_levels.ndim == 0:
                coords["level"] = [desiredlev]
            else:
                coords["level"] = requested_levels
        elif levels.ndim == 2:
            coords["level"] = (field3d.dims[-2:], requested_levels)
        else:
            if multiproduct:
                level_dims = field3d.dims[1:-3] + field3d.dims[-2:]
            else:
                level_dims = field3d.dims[:-3] + field3d.dims[-2:]
            coords["level"] = (level_dims, requested_levels)

        attrs.update(field3d.attrs)
        name = f"{field3d.name}_interp"

    vert_units = None
    if isinstance(vert, xr.DataArray):
        vert_units = vert.attrs.get("units")

    attrs["missing_value"] = missing
    attrs["_FillValue"] = missing
    attrs["vert_units"] = vert_units
    attrs.pop("MemoryOrder", None)
    attrs.pop("description", None)

    data_array = xr.DataArray(
        result,
        name=name,
        dims=dims,
        coords=coords,
        attrs=attrs,
    )
    return data_array.squeeze() if squeeze else data_array


def _runner_safe_masked_fallback(masked, squeeze):
    """Keep masks while preserving the shim's historical NaN array coercion."""
    fallback = masked
    if np.ma.is_masked(fallback) and np.issubdtype(fallback.dtype, np.inexact):
        fallback.data[np.ma.getmaskarray(fallback)] = np.nan
    return fallback.squeeze() if squeeze else fallback


def _interplevel_legacy(field3d, vert, desiredlev):
    """Return the established wrf-rust 0.2.35 three-argument result.

    This path intentionally preserves the original public contract used by
    WRF-Runner: exactly three-dimensional inputs, scalar or two-dimensional
    targets, float64 ndarray output, NaN missing values, and logarithmic
    interpolation when the supplied vertical coordinate decreases upward.
    """
    field = np.asarray(field3d, dtype=np.float64)
    coordinate = np.asarray(vert, dtype=np.float64)

    if field.ndim != 3 or coordinate.ndim != 3:
        raise ValueError(
            "field_3d and vert_coord_3d must be 3-D arrays (nz, ny, nx)"
        )
    if field.shape != coordinate.shape:
        raise ValueError(
            f"Shape mismatch: field_3d {field.shape} vs "
            f"vert_coord_3d {coordinate.shape}"
        )

    nz, ny, nx = field.shape
    target_array = np.asarray(desiredlev, dtype=np.float64)
    if target_array.ndim == 0:
        target = np.full((ny, nx), float(target_array))
    elif target_array.ndim == 2:
        if target_array.shape != (ny, nx):
            raise ValueError(
                f"2D target_level shape {target_array.shape} doesn't match "
                f"field shape ({ny}, {nx})"
            )
        target = target_array
    else:
        raise ValueError("target_level must be a scalar or 2D array (ny, nx)")

    mid_j, mid_i = ny // 2, nx // 2
    descending = coordinate[0, mid_j, mid_i] > coordinate[-1, mid_j, mid_i]
    result = np.full((ny, nx), np.nan, dtype=np.float64)

    if descending:
        log_coordinate = np.log(np.clip(coordinate, 1.0e-10, None))
        log_target = np.log(target)
        for k in range(nz - 1):
            matches = (
                (coordinate[k] >= target)
                & (coordinate[k + 1] <= target)
                & np.isnan(result)
            )
            if not np.any(matches):
                continue

            denominator = log_coordinate[k + 1] - log_coordinate[k]
            safe_denominator = np.where(
                np.abs(denominator) < 1.0e-12, 1.0, denominator
            )
            fraction = (log_target - log_coordinate[k]) / safe_denominator
            interpolated = field[k] + fraction * (field[k + 1] - field[k])
            result = np.where(matches, interpolated, result)
    else:
        for k in range(nz - 1):
            matches = (
                (coordinate[k] <= target)
                & (coordinate[k + 1] >= target)
                & np.isnan(result)
            )
            if not np.any(matches):
                continue

            denominator = coordinate[k + 1] - coordinate[k]
            safe_denominator = np.where(
                np.abs(denominator) < 1.0e-12, 1.0, denominator
            )
            fraction = (target - coordinate[k]) / safe_denominator
            interpolated = field[k] + fraction * (field[k + 1] - field[k])
            result = np.where(matches, interpolated, result)

    return result


def interplevel(
    field3d,
    vert,
    desiredlev,
    missing=DEFAULT_FILL_FLOAT64,
    squeeze=True,
    meta=None,
):
    """Interpolate a field to one or more surfaces in a vertical coordinate.

    The explicit ``meta`` modes follow NCAR wrf-python 1.3.4.1 commit
    ``31c923335227b22fa656fd589a5342b91103e939`` ``interplevel`` semantics.
    In those modes, the rightmost input dimensions are ``(nz, ny, nx)``;
    arbitrary matching left dimensions are supported. ``field3d`` may
    additionally have one leading product dimension, as used by vector
    diagnostics such as ``wspd_wdir``.

    The extended ``desiredlev`` may be a scalar, a one-dimensional level
    sequence, one shared ``(ny, nx)`` target surface, or a target surface with
    left dimensions matching ``vert``. Interpolation is linear in ``vert``
    itself. Out-of-range columns are returned as masked values using
    ``missing``.

    Omitting ``meta`` preserves wrf-rust 0.2.35's three-argument contract: a
    float64 NumPy ndarray, NaN missing values, scalar or 2-D target levels, and
    logarithmic interpolation for descending pressure coordinates. Pass
    ``meta=True`` or ``meta=False`` explicitly to select the extended strict
    wrf-python-compatible path. In that path, ``meta=True`` returns an xarray
    DataArray when xarray is available, while ``meta=False`` returns a NumPy
    masked array.
    """
    if meta is None:
        return _interplevel_legacy(field3d, vert, desiredlev)

    field, coordinate, multiproduct = _validate_inputs(field3d, vert)
    levels, levels_are_surfaces = _normalize_levels(desiredlev, coordinate.shape)

    missing_array = np.asarray(missing)
    if missing_array.ndim != 0:
        raise TypeError("missing must be a scalar")
    try:
        missing_value = float(missing_array)
    except (TypeError, ValueError) as exc:
        raise TypeError("missing must be convertible to a floating-point value") from exc

    output = _interpolate_all(
        field,
        coordinate,
        levels,
        levels_are_surfaces,
        missing_value,
    )
    masked = np.ma.masked_values(output, missing_value)

    use_meta = bool(meta)
    if use_meta:
        xr = _load_xarray()
        if xr is not None:
            return _metadata_result(
                masked,
                field3d,
                vert,
                desiredlev,
                levels,
                levels_are_surfaces,
                missing,
                squeeze,
                multiproduct,
                xr,
            )
        return _runner_safe_masked_fallback(masked, squeeze)

    return masked.squeeze() if squeeze else masked


__all__ = ["DEFAULT_FILL_FLOAT64", "interplevel"]
