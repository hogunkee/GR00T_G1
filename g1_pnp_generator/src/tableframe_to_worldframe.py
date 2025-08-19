#!/usr/bin/env python3
"""
Table Frame to World Frame Coordinate Converter
테이블 좌표(x: [-0.9, 0.9], y: [-0.9, 0.9])를 월드 좌표로 변환.
최소제곱법으로 캘리브레이션 포인트에서 스케일과 센터 계산.
"""

from typing import Tuple
import numpy as np

# 캘리브레이션 포인트 (테이블 좌표 -> 월드 좌표)
_CALIBRATION_POINTS = {
    (0.0, 0.0): (2.6500094, -1.6500286, 0.96647376),
    (0.9, 0.0): (3.1810093, -1.6500287, 0.96647376),
    (0.0, 0.9): (2.6500094, -1.3305285, 0.96647376),
    (-0.9, 0.0): (2.1190093, -1.6500286, 0.96647376),
    (0.0, -0.9): (2.6500094, -1.9695289, 0.96647364),
    (0.9, 0.9): (3.1810093, -1.3305286, 0.96647376),
    (0.9, -0.9): (3.1810093, -1.9695286, 0.96647376),
    (-0.9, 0.9): (2.1190095, -1.3305285, 0.96647376),
    (-0.9, -0.9): (2.1190093, -1.9695287, 0.9664737),
    (0.45, 0.45): (2.9155092, -1.4902787, 0.9664737),
    (0.45, -0.45): (2.9155092, -1.8097787, 0.9664737),
    (-0.45, 0.45): (2.3845093, -1.4902786, 0.9664737),
    (-0.45, -0.45): (2.3845093, -1.8097786, 0.96647376),
}

# 최소제곱법으로 변환 파라미터 계산
def _calculate_transform():
    table_coords = np.array(list(_CALIBRATION_POINTS.keys()))  # (N, 2)
    world_coords = np.array([p[:2] for p in _CALIBRATION_POINTS.values()])  # (N, 2)
    
    # 최소제곱법: world = center + scale * table
    A = np.hstack([np.ones((len(table_coords), 1)), table_coords])  # [1, x, y]
    center_x, x_scale, _ = np.linalg.lstsq(A, world_coords[:, 0], rcond=None)[0]
    center_y, _, y_scale = np.linalg.lstsq(A, world_coords[:, 1], rcond=None)[0]
    
    center_world = (center_x, center_y)
    # z는 평균값
    z_height = np.mean([p[2] for p in _CALIBRATION_POINTS.values()])
    
    return center_world, x_scale, y_scale, z_height

# 변환 파라미터 계산
_center_world, _x_scale, _y_scale, _z_height = _calculate_transform()

def table_to_world(table_pos: Tuple[float, float], height_offset: float = 0.0) -> Tuple[float, float, float]:
    """
    테이블 좌표를 월드 좌표로 변환.
    
    Args:
        table_pos: (x, y) 테이블 좌표 (x: [-0.9, 0.9], y: [-0.9, 0.9])
        height_offset: 테이블 위 추가 높이 (기본: 0.0)
    
    Returns:
        (x, y, z) 월드 좌표
    """
    table_x, table_y = table_pos
    
    # 범위 검증
    if not (-0.9 <= table_x <= 0.9):
        print(f"경고: 테이블 x 좌표 {table_x}가 [-0.9, 0.9] 범위를 벗어났습니다.")
    if not (-0.9 <= table_y <= 0.9):
        print(f"경고: 테이블 y 좌표 {table_y}가 [-0.9, 0.9] 범위를 벗어났습니다.")
    
    # 선형 변환
    world_x = _center_world[0] + _x_scale * table_x
    world_y = _center_world[1] + _y_scale * table_y
    world_z = _z_height + height_offset
    
    return (world_x, world_y, world_z)

def world_to_table(world_pos: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    월드 좌표를 테이블 좌표로 변환.
    
    Args:
        world_pos: (x, y, z) 월드 좌표
    
    Returns:
        (x, y) 테이블 좌표
    """
    world_x, world_y, _ = world_pos
    
    # 역변환
    table_x = (world_x - _center_world[0]) / _x_scale
    table_y = (world_y - _center_world[1]) / _y_scale
    
    return (table_x, table_y)

# 테스트
if __name__ == "__main__":
    print("=== 테이블 좌표 -> 월드 좌표 테스트 ===")
    test_cases = [(0.3,0.3)]
    for table_pos in test_cases:
        world = table_to_world(table_pos)
        table_back = world_to_table(world)
        print(f"테이블 {table_pos} -> 월드 ({world[0]:.7f}, {world[1]:.7f}, {world[2]:.7f}) -> 테이블 ({table_back[0]:.7f}, {table_back[1]:.7f})")