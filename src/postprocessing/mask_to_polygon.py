"""마스크를 폴리곤으로 변환"""
import numpy as np
import cv2
from typing import List, Optional, Any, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate

from ..utils.logger import logger


class MaskToPolygonConverter:
    """마스크 이미지를 폴리곤으로 변환"""
    
    def __init__(
        self,
        simplification_tolerance: float = 1.0,
        min_polygon_area: float = 100.0,
    ):
        """
        Args:
            simplification_tolerance: Douglas-Peucker 단순화 공차 (픽셀)
            min_polygon_area: 최소 폴리곤 면적 (픽셀^2)
        """
        self.simplification_tolerance = simplification_tolerance
        self.min_polygon_area = min_polygon_area
    
    def mask_to_polygon(
        self,
        mask: Any,
    ) -> Optional[Union[Polygon, MultiPolygon]]:
        """단일 마스크를 폴리곤으로 변환
        
        Args:
            mask: 이진 마스크 (H, W), 값 > 0인 영역이 객체
            
        Returns:
            shapely Polygon (또는 MultiPolygon), 유효하지 않으면 None
        """
        if isinstance(mask, dict):
            polygon = self.mask_to_polygon(mask['mask'])
            if polygon is None:
                return None
            return translate(
                polygon,
                xoff=float(mask.get('offset_x', 0)),
                yoff=float(mask.get('offset_y', 0)),
            )
        
        mask = np.asarray(mask)
        
        # 마스크 이진화
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 외곽선 검출 (OpenCV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            logger.warning("No contours found in mask")
            return None
        
        # 가장 큰 외곽선 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 외곽선을 폴리곤으로 변환
        polygon = self._contour_to_polygon(largest_contour)
        
        if polygon is None or polygon.area < self.min_polygon_area:
            logger.warning(f"Polygon area ({polygon.area if polygon else 0}) is below minimum")
            return None
        
        return polygon
    
    def masks_to_polygons(
        self,
        masks: List[Any],
    ) -> List[Union[Polygon, MultiPolygon]]:
        """여러 마스크를 폴리곤으로 변환
        
        Args:
            masks: 마스크 리스트
            
        Returns:
            폴리곤 리스트 (None 제외)
        """
        polygons = []
        for i, mask in enumerate(masks):
            try:
                polygon = self.mask_to_polygon(mask)
                if polygon is not None:
                    polygons.extend(self._explode_geometry(polygon))
            except Exception as e:
                logger.warning(f"Failed to convert mask {i}: {e}")
        
        logger.info(f"Converted {len(polygons)}/{len(masks)} masks to polygons")
        return polygons
    
    def _contour_to_polygon(self, contour: np.ndarray) -> Optional[Polygon]:
        """OpenCV 외곽선을 shapely Polygon으로 변환
        
        Args:
            contour: OpenCV 외곽선 (N, 1, 2)
            
        Returns:
            shapely Polygon
        """
        # 좌표 추출
        coords = contour.squeeze()
        
        if len(coords) < 3:
            return None

        # (x, y) 좌표로 변환
        if len(coords.shape) == 1:
            coords = coords.reshape(-1, 2)
        
        # Douglas-Peucker 단순화 적용
        coords = self._simplify_contour(coords)
        
        if len(coords) < 3:
            return None
        
        try:
            # 폴리곤 생성
            polygon = Polygon(coords)
            
            # 유효성 검사
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # 자-교차 제거
            
            if polygon.is_empty or polygon.area == 0:
                return None
            
            return polygon
        except Exception as e:
            logger.warning(f"Failed to create polygon: {e}")
            return None

    def _explode_geometry(
        self,
        geometry: Union[Polygon, MultiPolygon],
    ) -> List[Polygon]:
        """MultiPolygon을 개별 Polygon으로 펼치고 작은 조각은 제거"""
        if isinstance(geometry, Polygon):
            return [geometry] if geometry.area >= self.min_polygon_area else []
        
        if isinstance(geometry, MultiPolygon):
            polygons = [
                geom
                for geom in geometry.geoms
                if isinstance(geom, Polygon) and geom.area >= self.min_polygon_area
            ]
            return polygons
        
        logger.warning(f"Unsupported geometry type: {geometry.geom_type}")
        return []
    
    def _simplify_contour(
        self,
        coords: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> np.ndarray:
        """Douglas-Peucker 알고리즘으로 외곽선 단순화
        
        Args:
            coords: 좌표 배열 (N, 2)
            epsilon: 단순화 공차 (None이면 self.simplification_tolerance 사용)
            
        Returns:
            단순화된 좌표 배열 (M, 2), M <= N
        """
        if epsilon is None:
            epsilon = self.simplification_tolerance
        
        if len(coords) < 3:
            return coords
        
        # OpenCV 실제 외곽선 형식으로 변환
        contour_cv = coords.astype(np.float32).reshape(-1, 1, 2)
        
        # Douglas-Peucker 단순화
        simplified = cv2.approxPolyDP(contour_cv, epsilon, closed=True)
        
        return simplified.squeeze()
    
    def filter_small_polygons(
        self,
        polygons: List[Union[Polygon, MultiPolygon]],
        min_area: Optional[float] = None,
    ) -> List[Polygon]:
        """작은 폴리곤 필터링
        
        Args:
            polygons: 폴리곤 리스트
            min_area: 최소 면적 (None이면 self.min_polygon_area 사용)
            
        Returns:
            필터링된 폴리곤 리스트
        """
        if min_area is None:
            min_area = self.min_polygon_area
        
        filtered = []
        for polygon in polygons:
            if isinstance(polygon, MultiPolygon):
                filtered.extend([p for p in polygon.geoms if p.area >= min_area])
            elif polygon.area >= min_area:
                filtered.append(polygon)
        logger.info(f"Filtered: {len(polygons)} -> {len(filtered)} polygons")
        
        return filtered


class RightAngleRegularizer:
    """폴리곤의 직각을 정규화"""
    
    def __init__(
        self,
        tolerance_degrees: float = 5.0,
        snap_to_grid: bool = False,
        grid_size: float = 1.0,
    ):
        """
        Args:
            tolerance_degrees: 직각 인식 공차 (도)
            snap_to_grid: 그리드에 스냅 여부
            grid_size: 그리드 크기 (픽셀)
        """
        self.tolerance_degrees = tolerance_degrees
        self.snap_to_grid = snap_to_grid
        self.grid_size = grid_size
    
    def regularize_polygon(self, polygon: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon]:
        """폴리곤의 직각을 정규화
        
        Args:
            polygon: 입력 폴리곤
            
        Returns:
            정규화된 폴리곤
        """
        if isinstance(polygon, MultiPolygon):
            regularized_parts = [
                self.regularize_polygon(part)
                for part in polygon.geoms
                if isinstance(part, Polygon) and not part.is_empty
            ]
            return MultiPolygon(regularized_parts) if regularized_parts else polygon
        
        coords = np.array(polygon.exterior.coords[:-1])  # 마지막 점 제외
        
        # 각 꼭짓점 정규화
        regularized_coords = self._regularize_vertices(coords)
        
        if self.snap_to_grid:
            regularized_coords = self._snap_to_grid(regularized_coords)
        
        # 폴리곤 재생성
        try:
            new_polygon = Polygon(regularized_coords)
            if new_polygon.is_valid:
                return new_polygon
        except:
            pass
        
        return polygon
    
    def _regularize_vertices(self, coords: np.ndarray) -> np.ndarray:
        """각 꼭짓점의 각도를 직각 근처로 정규화
        
        Args:
            coords: 좌표 배열 (N, 2)
            
        Returns:
            정규화된 좌표 배열
        """
        n = len(coords)
        regularized = coords.copy()
        
        for i in range(n):
            prev_point = coords[(i - 1) % n]
            curr_point = coords[i]
            next_point = coords[(i + 1) % n]
            
            # 벡터 계산
            v1 = prev_point - curr_point
            v2 = next_point - curr_point
            
            # 각도 계산
            angle = self._calculate_angle(v1, v2)
            
            # 가장 가까운 직각 (0, 90, 180, 270도)으로 정렬
            nearest_right_angle = self._find_nearest_right_angle(angle)
            
            if abs(angle - nearest_right_angle) < self.tolerance_degrees:
                # 직각으로 조정
                new_direction = nearest_right_angle * np.pi / 180
                # 다음 점의 방향으로 조정
                distance = np.linalg.norm(v2)
                regularized[(i + 1) % n] = curr_point + np.array([
                    distance * np.cos(new_direction),
                    distance * np.sin(new_direction),
                ])
        
        return regularized
    
    @staticmethod
    def _calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        """두 벡터 사이의 각도 계산
        
        Returns:
            각도 (도)
        """
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    @staticmethod
    def _find_nearest_right_angle(angle: float) -> float:
        """가장 가까운 직각 찾기
        
        Args:
            angle: 각도 (도)
            
        Returns:
            가장 가까운 직각 (0, 90, 180, 270)
        """
        candidates = [0, 90, 180, 270]
        return min(candidates, key=lambda x: abs(angle - x))
    
    def _snap_to_grid(self, coords: np.ndarray) -> np.ndarray:
        """좌표를 그리드에 스냅
        
        Args:
            coords: 좌표 배열 (N, 2)
            
        Returns:
            스냅된 좌표
        """
        return np.round(coords / self.grid_size) * self.grid_size
    
    def regularize_polygons(self, polygons: List[Union[Polygon, MultiPolygon]]) -> List[Union[Polygon, MultiPolygon]]:
        """여러 폴리곤 정규화"""
        regularized = []
        for polygon in polygons:
            try:
                reg = self.regularize_polygon(polygon)
                regularized.append(reg)
            except Exception as e:
                logger.warning(f"Failed to regularize polygon: {e}")
                regularized.append(polygon)
        
        return regularized
