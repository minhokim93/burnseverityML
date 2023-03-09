
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rx
from multiprocessing import Pool, cpu_count
from shapely.ops import unary_union
 
 
def union(x):
    return unary_union(x)
 
if __name__ == "__main__":
    print("Reading raster...")
    dem = rx.open_rasterio('/Users/MacBookPro/Documents/Berkeley/ExplainableFire/doi_10.6071_M3QX18__v7/burn_severity_CA/ca3268211688619840421_burn_severity_CA.tif')
    fid = 'ca3268211688619840421'
    x, y, elevation = dem.x.values, dem.y.values, dem.values
    x, y = np.meshgrid(x, y)
    x, y, elevation = x.flatten(), y.flatten(), elevation.flatten()
 
    print("Converting to GeoDataFrame...")
    dem_pd = pd.DataFrame.from_dict({'severity': elevation, 'x': x, 'y': y})
    severity = 4
    dem_pd = dem_pd[dem_pd['severity'].isin([1,2,3,4,5])]
    dem_vector = gpd.GeoDataFrame(geometry=gpd.points_from_xy(dem_pd['x'], dem_pd['y'], crs=dem.rio.crs))
    dem_vector = dem_vector.buffer(15, cap_style=3)
    dem_vector = dem_vector.to_crs('EPSG:4326')
    geom_arr = []
 
    # Converting GeoSeries to list of geometries
    geoms = list(dem_vector)
 
    # Converting geometries list to nested list of geometries
    for i in range(0, len(geoms), 10000):
        geom_arr.append(geoms[i:i+10000])
 
    # Creating multiprocessing pool to perform union operation of chunks of geometries
    with Pool(cpu_count()) as p:
        geom_union = p.map(union, geom_arr)
 
    # Perform union operation on returned unioned geometries
    total_union = unary_union(geom_union)
 
    # Creating GeoDataFrame for total_union
    union_vector_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(total_union))
    union_vector_gdf.to_file(f"""burn_{fid}.gpkg""", crs='EPSG:4326', driver='GPKG')