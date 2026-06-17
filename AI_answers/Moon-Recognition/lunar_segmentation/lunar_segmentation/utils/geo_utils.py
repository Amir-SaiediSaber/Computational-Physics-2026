import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from pyproj import CRS, Transformer
from pathlib import Path
import numpy as np

# Moon radius in meters
MOON_RADIUS_M = 1737400.0
MOON_GEOG_CRS = CRS.from_proj4(f'+proj=longlat +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +no_defs')

def crop_singleband_raster(raster_path: Path, bounds):
    """
    Crop a single-band raster to the given (min_lon, min_lat, max_lon, max_lat) bounds.
    Bounds are in WGS84 degrees (EPSG:4326).

    For Moon imagery in EPSG:9122 (Equirectangular Moon with meters),
    we need to convert from WGS84 to the raster's coordinate system.
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    with rasterio.open(raster_path) as ds:
        raster_crs = ds.crs

        # Check if raster uses Equirectangular Moon (EPSG:9122 or similar)
        # These have units in meters, not degrees
        crs_dict = raster_crs.to_dict() if hasattr(raster_crs, 'to_dict') else {}
        units = crs_dict.get('units', 'm')

        if units != 'm':
            # Raster uses degrees, convert bounds directly
            native_bounds = (min_lon, min_lat, max_lon, max_lat)
        else:
            # Raster uses meters - try to convert from WGS84
            # For Moon projections, this may fail due to celestial body mismatch
            try:
                native_bounds = transform_bounds(MOON_GEOG_CRS, raster_crs, min_lon, min_lat, max_lon, max_lat)
            except Exception:
                # If projection fails, warn and use raster's native bounds directly
                import warnings
                warnings.warn("CRS transformation failed, using raster's native bounds")
                native_bounds = ds.bounds

        window = from_bounds(*native_bounds, transform=ds.transform).round_offsets().round_lengths()

        # Handle cases where window might be out of bounds or empty
        if window.width <= 0 or window.height <= 0:
            # Return empty arrays with correct channel dim if crop is invalid
            return np.zeros((0, 0), dtype=ds.dtypes[0]), ds.transform, ds.profile

        img = ds.read(1, window=window)
        transform = ds.window_transform(window)
        profile = ds.profile.copy()
        profile.update(height=img.shape[0], width=img.shape[1], transform=transform)

    return img, transform, profile

