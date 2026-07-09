"""Analytic WRF latitude/longitude and grid-coordinate transforms.

The equations in this module are a direct Python transcription of NCAR
wrf-python 1.3.4.1's ``DLLTOIJ`` and ``DIJTOLL`` routines at commit
31c923335227b22fa656fd589a5342b91103e939.  WRF uses a spherical earth with
a radius of 6,370,000 metres for these routines.
"""

from dataclasses import dataclass
import math


WRF_EARTH_RADIUS = 6_370_000.0
_RAD_PER_DEG = math.pi / 180.0
_DEG_PER_RAD = 180.0 / math.pi


@dataclass(frozen=True)
class ProjectionParams:
    """Projection metadata and the zero-based location of a known point."""

    map_proj: int
    truelat1: float
    truelat2: float
    stand_lon: float
    ref_lat: float
    ref_lon: float
    dx: float
    dy: float
    pole_lat: float = 90.0
    pole_lon: float = 0.0
    known_x: float = 0.0
    known_y: float = 0.0
    latinc: float = 0.0
    loninc: float = 0.0

    def __post_init__(self):
        values = (
            self.truelat1,
            self.truelat2,
            self.stand_lon,
            self.ref_lat,
            self.ref_lon,
            self.dx,
            self.dy,
            self.pole_lat,
            self.pole_lon,
            self.known_x,
            self.known_y,
            self.latinc,
            self.loninc,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("WRF projection parameters must be finite")
        if self.map_proj not in (1, 2, 3, 6):
            raise ValueError(
                f"unsupported WRF MAP_PROJ={self.map_proj}; expected 1, 2, 3, or 6"
            )
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("WRF DX and DY must be positive")
        if not -90.0 <= self.ref_lat <= 90.0:
            raise ValueError("WRF reference latitude must be in [-90, 90]")
        if self.map_proj == 6 and (self.latinc == 0.0 or self.loninc == 0.0):
            raise ValueError("WRF MAP_PROJ=6 requires non-zero LATINC and LONINC")


def normalize_longitude(longitude):
    """Normalize a longitude to wrf-python's [-180, 180] convention."""
    longitude = float(longitude)
    if not math.isfinite(longitude):
        raise ValueError("longitude must be finite")
    normalized = math.fmod(longitude, 360.0)
    if normalized > 180.0:
        normalized -= 360.0
    elif normalized < -180.0:
        normalized += 360.0
    return normalized


def longitude_delta(longitude, reference):
    """Return the shortest signed longitude difference in degrees."""
    delta = math.fmod(float(longitude) - float(reference), 360.0)
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    return delta


def _validate_latlon(latitude, longitude):
    latitude = float(latitude)
    longitude = float(longitude)
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("latitude and longitude must be finite")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be in [-90, 90]")
    return latitude, longitude


def _validate_xy(x, y):
    x = float(x)
    y = float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("x and y must be finite")
    return x, y


def _mercator_log_tan(latitude):
    if not -90.0 < latitude < 90.0:
        raise ValueError("Mercator coordinates require latitude strictly inside (-90, 90)")
    return math.log(math.tan(0.5 * ((latitude + 90.0) * _RAD_PER_DEG)))


def _polar_geometry(params):
    hemi = -1.0 if params.truelat1 < 0.0 else 1.0
    rebydx = WRF_EARTH_RADIUS / params.dx
    reflon = params.stand_lon + 90.0
    scale_top = 1.0 + hemi * math.sin(params.truelat1 * _RAD_PER_DEG)
    ala1 = params.ref_lat * _RAD_PER_DEG
    denominator = 1.0 + hemi * math.sin(ala1)
    if abs(denominator) < 1.0e-15:
        raise ValueError("polar stereographic reference point is at the opposite pole")
    rsw = rebydx * math.cos(ala1) * scale_top / denominator
    alo1 = (params.ref_lon - reflon) * _RAD_PER_DEG
    pole_x = params.known_x - rsw * math.cos(alo1)
    pole_y = params.known_y - hemi * rsw * math.sin(alo1)
    return hemi, rebydx, reflon, scale_top, pole_x, pole_y


def _lambert_geometry(params):
    truelat1 = params.truelat1
    truelat2 = params.truelat2
    if abs(truelat2) > 90.0:
        truelat2 = truelat1
    if abs(truelat1) >= 90.0 or abs(truelat2) >= 90.0:
        raise ValueError("Lambert true latitudes must be strictly inside (-90, 90)")

    if abs(truelat1 - truelat2) > 0.1:
        numerator = (
            math.log(math.cos(truelat1 * _RAD_PER_DEG))
            - math.log(math.cos(truelat2 * _RAD_PER_DEG))
        )
        denominator = (
            math.log(math.tan((90.0 - abs(truelat1)) * _RAD_PER_DEG * 0.5))
            - math.log(math.tan((90.0 - abs(truelat2)) * _RAD_PER_DEG * 0.5))
        )
        if abs(denominator) < 1.0e-15:
            raise ValueError("invalid Lambert true-latitude pair")
        cone = numerator / denominator
    else:
        cone = math.sin(abs(truelat1) * _RAD_PER_DEG)
    if abs(cone) < 1.0e-15:
        raise ValueError("Lambert cone factor is zero")

    hemi = -1.0 if truelat1 < 0.0 else 1.0
    rebydx = WRF_EARTH_RADIUS / params.dx
    ctl1r = math.cos(truelat1 * _RAD_PER_DEG)
    origin_numerator = math.tan(
        (90.0 * hemi - params.ref_lat) * _RAD_PER_DEG * 0.5
    )
    origin_denominator = math.tan(
        (90.0 * hemi - truelat1) * _RAD_PER_DEG * 0.5
    )
    ratio = origin_numerator / origin_denominator
    if ratio <= 0.0:
        raise ValueError("Lambert reference point is outside the projection hemisphere")
    rsw = rebydx * ctl1r / cone * ratio**cone

    delta_lon = longitude_delta(params.ref_lon, params.stand_lon)
    arg = cone * delta_lon * _RAD_PER_DEG
    pole_x = hemi * params.known_x - hemi * rsw * math.sin(arg)
    pole_y = hemi * params.known_y + rsw * math.cos(arg)
    return hemi, cone, rebydx, ctl1r, pole_x, pole_y, truelat2


def _rotate_coords(latitude, longitude, pole_lat, pole_lon, stand_lon, direction):
    """Transcribe wrf-python's ROTATECOORDS helper."""
    phi_np = pole_lat * _RAD_PER_DEG
    lam_np = pole_lon * _RAD_PER_DEG
    lam_0 = stand_lon * _RAD_PER_DEG
    rlat = latitude * _RAD_PER_DEG
    rlon = longitude * _RAD_PER_DEG
    dlam = math.pi - lam_0 if direction < 0 else lam_np

    sinphi = (
        math.cos(phi_np) * math.cos(rlat) * math.cos(rlon - dlam)
        + math.sin(phi_np) * math.sin(rlat)
    )
    sinphi = max(-1.0, min(1.0, sinphi))
    cosphi = math.sqrt(max(0.0, 1.0 - sinphi * sinphi))
    coslam = (
        math.sin(phi_np) * math.cos(rlat) * math.cos(rlon - dlam)
        - math.cos(phi_np) * math.sin(rlat)
    )
    sinlam = math.cos(rlat) * math.sin(rlon - dlam)
    if cosphi != 0.0:
        coslam /= cosphi
        sinlam /= cosphi

    out_lat = _DEG_PER_RAD * math.asin(sinphi)
    out_lon = _DEG_PER_RAD * (
        math.atan2(sinlam, coslam) - dlam - lam_0 + lam_np
    )
    return out_lat, out_lon


def ll_to_xy(params, latitude, longitude):
    """Convert one geographic coordinate to zero-based fractional x/y."""
    latitude, longitude = _validate_latlon(latitude, longitude)

    if params.map_proj == 3:
        clain = math.cos(params.truelat1 * _RAD_PER_DEG)
        if abs(clain) < 1.0e-15:
            raise ValueError("Mercator TRUELAT1 cannot be a pole")
        dlon = params.dx / (WRF_EARTH_RADIUS * clain)
        rsw = 0.0
        if params.ref_lat != 0.0:
            rsw = _mercator_log_tan(params.ref_lat) / dlon
        x = params.known_x + longitude_delta(longitude, params.ref_lon) * _RAD_PER_DEG / dlon
        y = params.known_y + _mercator_log_tan(latitude) / dlon - rsw
        return x, y

    if params.map_proj == 2:
        hemi, rebydx, reflon, scale_top, pole_x, pole_y = _polar_geometry(params)
        ala = latitude * _RAD_PER_DEG
        denominator = 1.0 + hemi * math.sin(ala)
        if abs(denominator) < 1.0e-15:
            raise ValueError("polar stereographic target is at the opposite pole")
        radius = rebydx * math.cos(ala) * scale_top / denominator
        angle = (longitude - reflon) * _RAD_PER_DEG
        return (
            pole_x + radius * math.cos(angle),
            pole_y + hemi * radius * math.sin(angle),
        )

    if params.map_proj == 1:
        hemi, cone, rebydx, ctl1r, pole_x, pole_y, _ = _lambert_geometry(params)
        numerator = math.tan((90.0 * hemi - latitude) * _RAD_PER_DEG * 0.5)
        denominator = math.tan(
            (90.0 * hemi - params.truelat1) * _RAD_PER_DEG * 0.5
        )
        ratio = numerator / denominator
        if ratio <= 0.0:
            raise ValueError("Lambert target is outside the projection hemisphere")
        radius = rebydx * ctl1r / cone * ratio**cone
        angle = cone * longitude_delta(longitude, params.stand_lon) * _RAD_PER_DEG
        x = pole_x + hemi * radius * math.sin(angle)
        y = pole_y - radius * math.cos(angle)
        return hemi * x, hemi * y

    target_lat = latitude
    target_lon = longitude
    ref_lat = params.ref_lat
    ref_lon = params.ref_lon
    if params.pole_lat != 90.0:
        target_lat, rotated_lon = _rotate_coords(
            target_lat,
            target_lon,
            params.pole_lat,
            params.pole_lon,
            params.stand_lon,
            -1,
        )
        target_lon = rotated_lon + params.stand_lon
        ref_lat, rotated_ref_lon = _rotate_coords(
            ref_lat,
            ref_lon,
            params.pole_lat,
            params.pole_lon,
            params.stand_lon,
            -1,
        )
        ref_lon = rotated_ref_lon + params.stand_lon

    x = params.known_x + longitude_delta(target_lon, ref_lon) / params.loninc
    y = params.known_y + (target_lat - ref_lat) / params.latinc
    return x, y


def xy_to_ll(params, x, y):
    """Convert one zero-based fractional x/y coordinate to latitude/longitude."""
    x, y = _validate_xy(x, y)

    if params.map_proj == 3:
        clain = math.cos(params.truelat1 * _RAD_PER_DEG)
        if abs(clain) < 1.0e-15:
            raise ValueError("Mercator TRUELAT1 cannot be a pole")
        dlon = params.dx / (WRF_EARTH_RADIUS * clain)
        rsw = 0.0
        if params.ref_lat != 0.0:
            rsw = _mercator_log_tan(params.ref_lat) / dlon
        latitude = (
            2.0 * math.atan(math.exp(dlon * (rsw + y - params.known_y)))
            * _DEG_PER_RAD
            - 90.0
        )
        longitude = params.ref_lon + (x - params.known_x) * dlon * _DEG_PER_RAD
        return latitude, normalize_longitude(longitude)

    if params.map_proj == 2:
        hemi, rebydx, reflon, scale_top, pole_x, pole_y = _polar_geometry(params)
        xx = x - pole_x
        yy = (y - pole_y) * hemi
        radius_sq = xx * xx + yy * yy
        if radius_sq == 0.0:
            return hemi * 90.0, normalize_longitude(reflon)
        gi2 = (rebydx * scale_top) ** 2
        sine_lat = (gi2 - radius_sq) / (gi2 + radius_sq)
        sine_lat = max(-1.0, min(1.0, sine_lat))
        latitude = _DEG_PER_RAD * hemi * math.asin(sine_lat)
        longitude = reflon + _DEG_PER_RAD * math.atan2(yy, xx)
        return latitude, normalize_longitude(longitude)

    if params.map_proj == 1:
        hemi, cone, rebydx, _, pole_x, pole_y, truelat2 = _lambert_geometry(params)
        x_new = hemi * x
        y_new = hemi * y
        xx = x_new - pole_x
        yy = pole_y - y_new
        radius_sq = xx * xx + yy * yy
        if radius_sq == 0.0:
            return hemi * 90.0, normalize_longitude(params.stand_lon)

        radius = math.sqrt(radius_sq) / rebydx
        longitude = params.stand_lon + (
            _DEG_PER_RAD * math.atan2(hemi * xx, yy) / cone
        )
        chi1 = (90.0 - hemi * params.truelat1) * _RAD_PER_DEG
        chi2 = (90.0 - hemi * truelat2) * _RAD_PER_DEG
        if chi1 == chi2:
            power_base = radius / math.tan(chi1)
        else:
            power_base = radius * cone / math.sin(chi1)
        if power_base < 0.0:
            raise ValueError("Lambert x/y lies outside the invertible projection")
        chi = 2.0 * math.atan(
            power_base ** (1.0 / cone) * math.tan(chi1 * 0.5)
        )
        latitude = (90.0 - chi * _DEG_PER_RAD) * hemi
        return latitude, normalize_longitude(longitude)

    x_new = x - params.known_x
    y_new = y - params.known_y
    delta_lat = y_new * params.latinc
    delta_lon = x_new * params.loninc

    ref_lat = params.ref_lat
    ref_lon = params.ref_lon
    if params.pole_lat != 90.0:
        ref_lat, rotated_ref_lon = _rotate_coords(
            ref_lat,
            ref_lon,
            params.pole_lat,
            params.pole_lon,
            params.stand_lon,
            -1,
        )
        ref_lon = rotated_ref_lon + params.stand_lon

    latitude = delta_lat + ref_lat
    longitude = delta_lon + ref_lon
    if params.pole_lat != 90.0:
        latitude, longitude = _rotate_coords(
            latitude,
            longitude - params.stand_lon,
            params.pole_lat,
            params.pole_lon,
            params.stand_lon,
            1,
        )
    return latitude, normalize_longitude(longitude)
