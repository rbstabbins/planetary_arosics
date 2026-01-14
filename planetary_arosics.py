from pathlib import Path
from typing import Union, Tuple
import arosics
from arosics import DESHIFTER
from geoarray.baseclasses import GeoArray
import numpy as np
from osgeo import gdal
from pyproj import CRS, Transformer
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds


# Loading and helper functions

def generate_output_path(input_path: Union[Path,str], suffix: str) -> Path:
    """Generate an output file path based on the input file path.
    :param input_path: Path to the input file.
    :type input_path: Path
    :return: Path to the output file.
    :rtype: Path
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)

    input_dir = input_path.parent.parent.parent.parent

    instrument = input_path.parent.parent.parent.name
    product = input_path.parent.parent.name
    scene = input_path.parent.name
    output_dir = input_dir.parent / 'output_products' / instrument / product / scene
    output_dir.mkdir(parents=True, exist_ok=True)   
    
    # define the output file path    
    output_path = output_dir / (input_path.stem + suffix)    

    return output_path

def get_raster_bounds(
        raster_path: Union[str, Path], 
        bounds_type: Union[str, np.array]) -> Union[str, np.array]:
    """Get the bounds of a raster file.

    :param raster_path: Path to the raster file.
    :type raster_path: Union[str, Path]
    :return: Bounds of the raster file as a string.
    :rtype: str
    """
    with rio.open(raster_path) as src:
        bounds = np.array(src.bounds)
    if bounds_type is str:
        return f"{bounds[0]:.2f}/{bounds[2]:.2f}/{bounds[1]:.2f}/{bounds[3]:.2f}"
    else:
        return bounds
        
# def str_bounds_to_numpy(bounds_str: str) -> np.ndarray:
#     """Convert a bounds string to a numpy array.

#     :param bounds_str: Bounds string in the format "minx/maxx/miny/maxy".
#     :type bounds_str: str
#     :return: Numpy array of bounds [minx, miny, maxx, maxy].
#     :rtype: np.ndarray
#     """
#     bounds_list = [float(coord) for coord in bounds_str.split('/')]
#     return np.array([bounds_list[0], bounds_list[2], bounds_list[1], bounds_list[3]])

def pds2geotiff(pds_file: Path, noData: float=0) -> Path:
    """Convert a PDS .img file to a GeoTIFF file.

    :param pds_file: Path to the PDS .img file to be converted.
    :type pds_file: Path
    :return: Path to the converted GeoTIFF file.
    :rtype: Path
    """    
    gdal.AllRegister()

    geo_tiff_out = generate_output_path(pds_file, suffix='.tif')

    gdal.Translate(
        str(geo_tiff_out),
        str(pds_file),
        format='GTiff',        
        noData=noData,
        creationOptions=['COMPRESS=LZW']
    )

    return geo_tiff_out

# CRS conversion functions

def geotiff_2_geo_crs(gtiff_file: Path) -> Path:
    """Convert the CRS of a GeoTIFF file to a geographic CRS.

    The function assumes that the input file path has the structure:
    /input_products/<instrument>/<product>/<scene>/<filename>.tif

    such that the output product will be saved as:
    /output_products/<instrument>/<scene>/<filename>_grs.tif

    :param gtiff_file: Path to the GeoTIFF file.
    :type gtiff_file: Union[str, Path]
    :return: Path to the converted GeoTIFF file.
    :rtype: Union[str, Path]
    """
    # check if the input is a Path object
    geo_gtiff_file = generate_output_path(gtiff_file, suffix='_grs.tif')

    # open the GeoTIFF file with rasterio
    with rio.open(gtiff_file, driver='GTiff') as src:
        # check if the CRS is already geographic
        if src.crs.is_geographic:
            print(f"{gtiff_file.name} is already in geographic CRS.")
            return gtiff_file
        
        # create a new CRS object for geographic coordinates
        geo_crs = src.crs.from_wkt(CRS.from_wkt(src.crs.to_wkt()).geodetic_crs.to_wkt())
        
        # calculate the transform and dimensions for the new CRS
        transform, width, height = calculate_default_transform(
            src.crs, geo_crs, src.width, src.height, *src.bounds)
        
        # update the metadata with the new CRS and transform
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': geo_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # create a new GeoTIFF file with the updated CRS        
        with rio.open(geo_gtiff_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=geo_crs,
                    resampling=Resampling.nearest)

    return geo_gtiff_file

def inherit_crs(
    src_crs_path: Path,
    dst_crs_path: Path,
    crs_id: str
    ) -> Path:
    """Inherit the CRS from a source GeoTIFF file to a destination GeoTIFF file."""

    # First we access the data with rasterio
    src_raster = rio.open(src_crs_path)
    dst_raster = rio.open(dst_crs_path)

    # Now we extract the CRS WKT strings from the rasterio objects
    src_crs_wkt = src_raster.crs.to_wkt()
    src_crs = CRS.from_wkt(src_crs_wkt)

    dst_crs_wkt = dst_raster.crs.to_wkt()
    dst_crs = CRS.from_wkt(dst_crs_wkt)

    # Now we use the rasterio calculate_default_transform function and reproject 
    # to get the transform that changes the destination CRS to the source CRS.

    # calculate the transform and dimensions for the new CRS
    transform, width, height = calculate_default_transform(
        dst_crs, src_crs, dst_raster.width, dst_raster.height, *dst_raster.bounds)

    # update the metadata with the new CRS and transform
    kwargs = dst_raster.meta.copy()
    kwargs.update({
        'crs': src_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    # create a new GeoTIFF file with the updated CRS - need a new string to denote the change
    suffix = crs_id # need a concise name for the new CRS used)
    dst_crs_tmp_path = generate_output_path(dst_crs_path, suffix=f'_{suffix}.tif')

    with rio.open(dst_crs_tmp_path, 'w', **kwargs) as dst: # this has the properties of the CRISM, destination, raster
        for i in range(1, dst_raster.count + 1):
            # for each band in the destination crism raster, this reprojects the original 
            # destination raster (CRISM) to the new destination raster
            reproject(
                source=rio.band(dst_raster, i),
                destination=rio.band(dst, i),
                src_transform=src_raster.transform,
                src_crs=src_raster.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
        
    return dst_crs_tmp_path

# Co-registration functions

def bbox_mask_geoarray(geotiff_path, bbox, nodata_val=False):
    
    # """
    # Create a geoarray.GeoArray mask from a bounding box.

    # Pixels outside the bounding box are True (masked), inside are False.

    # Parameters
    # ----------
    # geotiff_path : str
    #     Path to a geographic GeoTIFF (north-up).
    # bbox : tuple
    #     (minx, miny, maxx, maxy) in same CRS as raster.
    # nodata_val : int or bool
    #     Pixel value for mask output (optional). Default False for inside bbox.

    # Returns
    # -------
    # mask_ga : geoarray.GeoArray
    #     A GeoArray whose `.arr` is a 2D boolean array,
    #     with geotransform + projection from source raster.
    # """

    # if bbox is string, convert to numpy array
    if isinstance(bbox, str):
        bbox = str_bounds_to_numpy(bbox)
        
    # Read raster metadata
    with rio.open(geotiff_path) as ds:
        # Compute the pixel window covering the bounding box
        window = from_bounds(*bbox, transform=ds.transform)
        window = window.round_offsets().round_lengths()

        # Make a base boolean array: True = outside bbox
        base_mask = np.ones((ds.height, ds.width), dtype=bool)

        # Set inside bbox pixels to False
        base_mask[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width
        ] = False

        # Optionally, represent masked vs. unmasked with nodata_val
        # (e.g., if users want 0/1 or specific class codes)
        mask_arr = base_mask.astype(type(nodata_val))
        mask_arr[~base_mask] = nodata_val

        # Create a GeoArray from this array & georeferencing info
        mask_ga = GeoArray(
            mask_arr,
            geotransform=ds.transform.to_gdal(),
            projection=ds.crs.to_wkt() if ds.crs else None
        )

    return mask_ga

def arosics_global_coreg(
    ref_path: Path,
    tgt_path: Path,
    no_data: int = 255,
    ws: int = 64,
    max_shift: int = 20,
    ref_mask_bbox: Union[None, tuple] = None,
    tgt_mask_bbox: Union[None, tuple] = None
    ) -> Tuple[Path, arosics.COREG]:
    """Perform global co-registration using AROSICS.

    :param ref_path: Path to the reference image.
    :type ref_path: Path
    :param tgt_path: Path to the target image.
    :type tgt_path: Path
    :param no_data: No data value for the images.
    :type no_data: int
    :param ws: Window size for the co-registration.
    :type ws: int
    :param max_shift: Maximum shift for the co-registration.
    :type max_shift: int
    :param ref_mask_bbox: Bounding box to mask the reference image (minx, miny, maxx, maxy).
    :type ref_mask_bbox: Union[None, tuple]       
    :param tgt_mask_bbox: Bounding box to mask the target image (minx, miny, maxx, maxy).
    :type tgt_mask_bbox: Union[None, tuple]
    :return: Path to the co-registered output image and the AROSICS COREg object.
    :rtype: Tuple[Path, arosics.COREG]
    """

    # set output path for the co-registered output
    coreg_path = generate_output_path(tgt_path, suffix='_crg.tif')

    # initialise target and/or reference bad data masks
    mask_baddata_ref = bbox_mask_geoarray(ref_path, ref_mask_bbox, nodata_val=False) if ref_mask_bbox else None
    mask_baddata_tgt = bbox_mask_geoarray(tgt_path, tgt_mask_bbox, nodata_val=False) if tgt_mask_bbox else None

    CRG = arosics.COREG(
        im_ref=str(ref_path),
        im_tgt=str(tgt_path),
        nodata=(no_data,no_data),
        ws=(ws,ws),
        max_shift=max_shift,
        path_out=str(coreg_path),
        mask_baddata_ref=mask_baddata_ref,
        mask_baddata_tgt=mask_baddata_tgt,
        fmt_out='GTIFF')
    
    CRG.calculate_spatial_shifts()
    
    CRG.correct_shifts()

    return coreg_path, CRG

def arosics_local_coreg(
    ref_path: Path,
    tgt_path: Path,
    grid_res: int = 8,
    no_data: int = 255,
    ws: int = 64,
    max_shift: int = 20,
    ):
    """Perform local co-registration using AROSICS.

    :param ref_path: Path to the reference image.
    :type ref_path: Path
    :param tgt_path: Path to the target image.
    :type tgt_path: Path
    :return: Path to the co-registered output image.
    :rtype: Path
    """
    # set output path for the co-registered output  
    local_coreg_path = generate_output_path(tgt_path, suffix='_crl.tif')

    CRL = arosics.COREG_LOCAL(
        im_ref=str(ref_path),
        im_tgt=str(tgt_path),
        grid_res=grid_res, # pixels separating tie points in tgt
        nodata=(no_data,no_data),
        window_size=(ws, ws),
        max_shift=max_shift,
        path_out=str(local_coreg_path),
        fmt_out='GTIFF')
    CRL.correct_shifts()

    return local_coreg_path, CRL

def apply_coreg(img2shift_path: Path, global_crg: None, local_crg: None) -> str:

    if not global_crg and not local_crg:
        raise ValueError("At least one of global_crg or local_crg must be provided.")

    if global_crg:
        global_shifted_img_path = generate_output_path(img2shift_path, suffix='_crg.tif')
        # first apply the global coregistration
        DESHIFTER(
            str(img2shift_path), 
            global_crg.coreg_info, 
            path_out=str(global_shifted_img_path), 
            fmt_out='GTIFF', 
            nodata=255).correct_shifts()
        shifted_img_path = global_shifted_img_path
    
    if global_crg and local_crg:
        img2shift_path = global_shifted_img_path
    
    if local_crg:
        # then apply the local coregistration
        local_shifted_img_path = generate_output_path(img2shift_path, suffix='_crl.tif')
        DESHIFTER(
            str(img2shift_path), 
            local_crg.coreg_info, 
            path_out=str(local_shifted_img_path), 
            fmt_out='GTIFF', 
            nodata=255).correct_shifts()
        shifted_img_path = local_shifted_img_path

    return shifted_img_path

# Visualization functions

# show raster

# show rasters.
