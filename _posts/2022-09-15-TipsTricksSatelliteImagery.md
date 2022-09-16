---
layout: post
title: Tips and Tricks for Working with Satellite Imagery
---

I am currently working with Google Earth Engine to do machine learning with satellite imagery. 
These are a couple of things I learned/wish I had known before pressing run on my code. 😅

## 1. Exporting and aligning images - `crs` and `crsTransform`
   
When using Google Earth Engine's Python API, you can export your satellite imagery to your Google Drive using [`Export.image.toDrive()`](https://developers.google.com/earth-engine/apidocs/export-image-todrive).

With the function, you can specify the image you want to download, as well as the region, size, and dimensions you want to download the image at. At a cursory glance, you may think that specifying the `region` (coordinates representing region to export) and `scale` (resolution in meters per pixel) arguments are enough to download an image. 
I initially did this and ignored the `crs` and `crsTransform` arguments. 
This is true, *only* if you are only looking at one image collections. 

If you want to compare two image collections, chances are the way their pixels are exported with respect to longitude and latitude will differ from each other. 
If you download images from these two image collections, each pixel will correspond to slightly different longitude and latitude values, even over the same region and scale.

The way an image collection organizes their pixels are shown by the `crs` and `transform` attributes. To get these attributes, for example for an image collection, `night_lights`, you can use `night_lights.projection().getInfo()`. This function returns this: 

`{'type': 'Projection', 'crs': 'EPSG:4326', 'transform': [1, 0, 0, 0, 1, 0]}`

The `crs` is the [Coordinate Reference System](https://en.wikipedia.org/wiki/Spatial_reference_system) and the `transform` refers to the transformation used to export the image. The numbers in the list represent [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation] as shown in the documentation [here](https://developers.google.com/earth-engine/guides/exporting). 

By choosing one `crs` and `transform` to use for all your image collection, you can export images that share the same longitude and latitude for each pixel. 

The code would look like this: 

{%highlight Python%}

import ee 

start_date = '2020-01-01'
end_date = '2020-12-31'

# night light data - get median image for year
night_light_2020 = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(start_date, end_date).select('avg_rad').median()

# census data
census_2020 = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Count").filterDate(start_date, end_date).first()

# get crs and crsTransform for the census data 
crs_census = census_2020.projection().getInfo()['crs']
transform_census = census_2020.projection().getInfo()['transform']

# specify region
region = ee.Geometry.BBox(135.9259, 35.4698, 140.8698 , 36.6422)

# export census image using census crs and transform 
task = ee.batch.Export.image.toDrive(image = census_2020,
                                     fileNamePrefix = 'censusexport',
                                     region = region,
                                     crsTransform=transform_census,
                                     crs = crs_census)

task.start()

# export night light image using census crs and transform to match lon/lat
task = ee.batch.Export.image.toDrive(image = night_light_2020,
                                     fileNamePrefix = 'nightlightexport',
                                     region = region,
                                     crsTransform=transform_census,
                                     crs = crs_census)

task.start()


{%endhighlight%}



## 2. You might not have to download or export the image...

If you don't actually need an image and just need a numpy array of the image values (which may be the case if you are doing machine learning), you can use the `geemap` package. 

The `geemap` package has a function [`ee_to_numpy()`](https://geemap.org/common/?h=ee_to_numpy#geemap.common.ee_to_geopandas) which can be used to turn a Google Earth Engine image region into a numpy array. 
The caveat is that it doesn't seem to include the `crs` or `crsTransform` arguments we talked about above so this may be exclusive if you are just working with one image collection. 


## 3. How to display your GeoTIFF 

I found that there are many ways you can show a GeoTIFF image - there are many packages out there, and you can also convert the image into a different datatype to plot it through `matplotlib` or `plotly`. 
Here are the easiest ways I found to plot different types of GeoTIFF images. 

### If you just have one band/gray-scale image: 

The easiest way I found was to use `rioxarray`. With `rioxarray`, plotting an image is as easy as two lines: 

{%highlight Python%}

image = rioxarray.open_rasterio('image path')
im.plot(cmap='gray')

{%endhighlight%}

<img src="/assets/tipstricksSI/img.png">

### If you have an RGB image: 

I found the [`earthpy`](https://earthpy.readthedocs.io/en/latest/) and `xarray` package to be helpful.

First, we need to create an `xarray` object of all the bands that we want to plot. 

{%highlight Python%}
import rioxarray

tiff = rioxarray.open_rasterio('../data/test_japan_landsat.tif')
tiff_ds = tiff.to_dataset('band')
# rename bands
tiff_ds = tiff_ds.rename({1: 'B1',
                          2: 'B2',
                          3: 'B3',
                          4: 'B4',
                          5: 'B5',
                          6: 'B6_VCID_1',
                          7: 'B6_VCID_2',
                          8: 'B7',
                          9: 'B8'})
tiff_ds
{%endhighlight%}

This creates an `xarray` Dataset that looks like this: 

![img.png](/assets/tipstricksSI/xarray_output.png)

Now, choose the bands we want to plot in RGB order using `xarray`: 

{%highlight Python%}
import xarray as xr

# for landsat, B3 = R, B2 = G, B1 = B
rgb_bands = [tiff_ds.B3, tiff_ds.B2, tiff_ds.B1]
# turn rgb bands into xarray object
landsat_rgb = xr.concat(rgb_bands, dim = "band")
{%endhighlight%}

And now, we can use the `earthpy` package: 

{%highlight Python%}
import earthpy.plot as ep
ep.plot_rgb(landsat_rgb.values)
plt.show()
{%endhighlight%}

![img.png](/assets/tipstricksSI/earthpy1.png)

This image looks very dark! We can explore this image using `ep.hist()`.
{%highlight Python%}
ep.hist(landsat_rgb.values, title = ["Band 3", "Band 2", "Band 1"])
plt.show()
{%endhighlight%}

![img.png](/assets/tipstricksSI/hist.png)

We can see that the values are very low since image values can take on values between 0 and 255. 

In order to be able to see the image better, we can use the `stretch` argument in `ep.plot_rgb()`. 

{%highlight Python%}
import earthpy.plot as ep
ep.plot_rgb(landsat_rgb.values, stretch=True)
plt.show()
{%endhighlight%}

We get the following image, that we are now able to see much better! 

![img.png](/assets/tipstricksSI/img_stretch.png)
