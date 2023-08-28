from osgeo import gdal
import os

def get_resolution(image_path):
    dataset = gdal.Open(image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    dataset = None
    return width, height, pixel_width, pixel_height

def calculate_resolution_in_meters(image_path):
    _, _, pixel_width, pixel_height = get_resolution(image_path)
    # 이미지의 픽셀 단위 크기를 미터로 변환하여 반환 (절댓값 사용)
    pixel_width_meters = abs(pixel_width)
    pixel_height_meters = abs(pixel_height)
    return pixel_width_meters, pixel_height_meters


image_path="./test_img/"
image_list=os.listdir(image_path)

for image in image_list:
    # 이미지 파일 경로를 입력하여 픽셀 당 미터 크기를 확인
    pixel_width_meters, pixel_height_meters = calculate_resolution_in_meters(image_path)
    print(f"Pixel Width: {pixel_width_meters} meters, Pixel Height: {pixel_height_meters} meters")