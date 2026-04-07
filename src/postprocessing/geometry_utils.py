"""지오메트리 유틸리티 및 벡터 파일 생성"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import transform as shapely_transform
import geopandas as gpd

from ..utils.logger import logger


class GeometryUtils:
    """지오메트리 관련 유틸리티"""
    
    @staticmethod
    def polygon_to_geojson_feature(
        polygon: Union[Polygon, MultiPolygon],
        properties: Dict[str, Any] = None,
        feature_id: int = 0,
    ) -> Dict[str, Any]:
        """폴리곤을 GeoJSON Feature로 변환
        
        Args:
            polygon: Shapely Polygon
            properties: 속성 딕셔너리
            feature_id: 피처 ID
            
        Returns:
            GeoJSON Feature JSON
        """
        if properties is None:
            properties = {}
        
        properties['id'] = feature_id
        properties['area'] = polygon.area
        properties['perimeter'] = polygon.length
        
        return {
            'type': 'Feature',
            'id': feature_id,
            'geometry': mapping(polygon),
            'properties': properties,
        }
    
    @staticmethod
    def polygon_to_coords_list(polygon: Union[Polygon, MultiPolygon]) -> Any:
        """폴리곤을 좌표 리스트로 변환"""
        if isinstance(polygon, MultiPolygon):
            return [
                list(part.exterior.coords)
                for part in polygon.geoms
                if isinstance(part, Polygon) and not part.is_empty
            ]
        return list(polygon.exterior.coords)

    @staticmethod
    def polygon_to_segmentation(polygon: Union[Polygon, MultiPolygon]) -> List[List[float]]:
        """COCO polygon segmentation 형식으로 변환"""
        if isinstance(polygon, MultiPolygon):
            return [
                np.asarray(part.exterior.coords).ravel().tolist()
                for part in polygon.geoms
                if isinstance(part, Polygon) and not part.is_empty
            ]
        return [np.asarray(polygon.exterior.coords).ravel().tolist()]
    
    @staticmethod
    def calculate_building_metrics(polygon: Union[Polygon, MultiPolygon]) -> Dict[str, float]:
        """건물 폴리곤의 메트릭 계산"""
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        area = polygon.area
        perimeter = polygon.length
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'width': float(width),
            'height': float(height),
            'bbox_aspect_ratio': float(width / height) if height > 0 else 0.0,
            'compactness': float(perimeter / (2 * np.sqrt(np.pi * area))) if area > 0 else 0.0,
        }

    @staticmethod
    def pixel_polygon_to_geo(
        polygon: Union[Polygon, MultiPolygon],
        affine_transform: Any,
    ) -> Union[Polygon, MultiPolygon]:
        """픽셀 좌표계 폴리곤을 GeoTIFF 원본 좌표계로 변환"""
        def _transform(x, y, z=None):
            pixel_x = np.asarray(x)
            pixel_y = np.asarray(y)
            geo_x = affine_transform.a * pixel_x + affine_transform.b * pixel_y + affine_transform.c
            geo_y = affine_transform.d * pixel_x + affine_transform.e * pixel_y + affine_transform.f
            if z is None:
                return geo_x, geo_y
            return geo_x, geo_y, z
        
        return shapely_transform(_transform, polygon)

    @staticmethod
    def pixel_polygons_to_geo(
        polygons: List[Union[Polygon, MultiPolygon]],
        affine_transform: Any,
    ) -> List[Union[Polygon, MultiPolygon]]:
        """픽셀 좌표계 폴리곤 목록을 GeoTIFF 원본 좌표계로 변환"""
        return [
            GeometryUtils.pixel_polygon_to_geo(polygon, affine_transform)
            for polygon in polygons
        ]


class VectorFileWriter:
    """벡터 파일 (GeoJSON, Shapefile) 생성"""
    
    def __init__(
        self,
        crs: str = "EPSG:4326",
        output_dir: str = "./outputs",
    ):
        """
        Args:
            crs: 좌표계 (예: EPSG:4326, EPSG:5186)
            output_dir: 출력 디렉토리
        """
        self.crs = crs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_geojson(
        self,
        polygons: List[Union[Polygon, MultiPolygon]],
        filename: str,
        properties_list: List[Dict] = None,
    ) -> str:
        """폴리곤들을 GeoJSON으로 저장
        
        Args:
            polygons: 폴리곤 리스트
            filename: 파일명 (확장자 제외)
            properties_list: 각 폴리곤의 속성 리스트
            
        Returns:
            저장된 파일 경로
        """
        output_path = self.output_dir / f"{filename}.geojson"
        
        features = []
        for i, polygon in enumerate(polygons):
            props = properties_list[i] if properties_list and i < len(properties_list) else {}
            feature = GeometryUtils.polygon_to_geojson_feature(
                polygon,
                properties=props,
                feature_id=i,
            )
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {'type': 'name', 'properties': {'name': self.crs}},
            'features': features,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GeoJSON saved: {output_path}")
        return str(output_path)
    
    def write_shapefile(
        self,
        polygons: List[Union[Polygon, MultiPolygon]],
        filename: str,
        properties_list: List[Dict] = None,
    ) -> str:
        """폴리곤들을 Shapefile로 저장
        
        Args:
            polygons: 폴리곤 리스트
            filename: 파일명 (확장자 제외)
            properties_list: 각 폴리곤의 속성 리스트
            
        Returns:
            저장된 파일 경로
        """
        output_path = self.output_dir / f"{filename}.shp"
        
        # GeoDataFrame 생성
        geometries = []
        properties_dicts = []
        
        for i, polygon in enumerate(polygons):
            geometries.append(polygon)
            props = properties_list[i] if properties_list and i < len(properties_list) else {}
            props['id'] = i
            props['area'] = polygon.area
            props['perimeter'] = polygon.length
            properties_dicts.append(props)
        
        gdf = gpd.GeoDataFrame(
            properties_dicts,
            geometry=geometries,
            crs=self.crs,
        )
        
        # Shapefile 저장
        gdf.to_file(output_path)
        logger.info(f"Shapefile saved: {output_path}")
        
        return str(output_path)
    
    def write_json(
        self,
        polygons: List[Union[Polygon, MultiPolygon]],
        filename: str,
        properties_list: List[Dict] = None,
    ) -> str:
        """폴리곤들을 JSON으로 저장 (좌표 리스트 형식)
        
        Args:
            polygons: 폴리곤 리스트
            filename: 파일명 (확장자 제외)
            properties_list: 각 폴리곤의 속성 리스트
            
        Returns:
            저장된 파일 경로
        """
        output_path = self.output_dir / f"{filename}.json"
        
        data = []
        for i, polygon in enumerate(polygons):
            item = {
                'id': i,
                'coordinates': GeometryUtils.polygon_to_coords_list(polygon),
                'metrics': GeometryUtils.calculate_building_metrics(polygon),
            }
            if properties_list and i < len(properties_list):
                item['properties'] = properties_list[i]
            data.append(item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON saved: {output_path}")
        return str(output_path)
    
    def write_coco_json(
        self,
        polygons: List[Union[Polygon, MultiPolygon]],
        image_path: str,
        filename: str,
    ) -> str:
        """COCO 형식의 JSON으로 저장"""
        output_path = self.output_dir / f"{filename}_coco.json"
        
        annotations = []
        for i, polygon in enumerate(polygons):
            # 폴리곤을 RLE 처럼 변환
            annotation = {
                'id': i,
                'image_id': 0,
                'category_id': 1,  # Building
                'area': polygon.area,
                'bbox': list(polygon.bounds),  # [x_min, y_min, x_max, y_max]
                'segmentation': GeometryUtils.polygon_to_segmentation(polygon),
                'iscrowd': 0,
            }
            annotations.append(annotation)
        
        coco_data = {
            'images': [{
                'id': 0,
                'file_name': image_path,
                'height': 0,  # 필요시 업데이트
                'width': 0,
            }],
            'annotations': annotations,
            'categories': [{
                'id': 1,
                'name': 'building',
                'supercategory': 'object',
            }],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"COCO JSON saved: {output_path}")
        return str(output_path)
