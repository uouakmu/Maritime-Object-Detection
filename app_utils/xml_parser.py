"""
XML 어노테이션 파싱 유틸리티
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple


def parse_xml_annotation(xml_path: str) -> Dict:
    """
    XML 어노테이션 파일을 파싱하여 객체 정보 추출
    
    Args:
        xml_path: XML 파일 경로
        
    Returns:
        Dict: {
            'filename': str,
            'size': {'width': int, 'height': int, 'depth': int},
            'objects': [
                {
                    'name': str,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'category_id': int
                },
                ...
            ]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 이미지 정보
    filename = root.find('filename').text if root.find('filename') is not None else ''
    
    size_elem = root.find('size')
    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)
    depth = int(size_elem.find('depth').text)
    
    # 객체 정보
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        category_id = int(obj.find('category_id').text)
        
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'category_id': category_id
        })
    
    return {
        'filename': filename,
        'size': {'width': width, 'height': height, 'depth': depth},
        'objects': objects
    }


def convert_to_yolo_format(bbox: List[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    바운딩 박스를 YOLO 형식으로 변환
    
    Args:
        bbox: [xmin, ymin, xmax, ymax]
        img_width: 이미지 너비
        img_height: 이미지 높이
        
    Returns:
        (x_center, y_center, width, height) - 모두 0~1 사이로 정규화
    """
    xmin, ymin, xmax, ymax = bbox
    
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymin) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return x_center, y_center, width, height


def get_annotation_files(annotation_dir: Path, date: str = None) -> List[Path]:
    """
    어노테이션 디렉토리에서 XML 파일 목록 가져오기
    
    Args:
        annotation_dir: 어노테이션 디렉토리 경로
        date: 특정 날짜 (예: '20201216'), None이면 전체
        
    Returns:
        XML 파일 경로 리스트
    """
    xml_files = []
    
    if date:
        date_dir = annotation_dir / date
        if date_dir.exists():
            xml_files = list(date_dir.rglob('*.xml'))
    else:
        xml_files = list(annotation_dir.rglob('*.xml'))
    
    return sorted(xml_files)


def get_image_path_from_annotation(xml_path: Path, image_base_dir: Path) -> Path:
    """
    XML 어노테이션 파일명으로부터 대응하는 이미지 파일 경로 찾기
    
    Args:
        xml_path: XML 파일 경로
        image_base_dir: 이미지 베이스 디렉토리
        
    Returns:
        이미지 파일 경로
    """
    # XML 파일명에서 이미지 파일명 추출
    # 예: 여수항_맑음_20201216_0014_0001.xml -> 여수항_맑음_20201216_0014_0001.jpg
    xml_filename = xml_path.stem  # 확장자 제외
    
    # XML 파일의 상대 경로 구조 유지
    # 예: 20201216/0014/여수항_맑음_20201216_0014_0001.xml
    relative_parts = xml_path.relative_to(xml_path.parents[2]).parts
    date = relative_parts[0]
    source_id = relative_parts[1]
    
    # 이미지 경로 구성
    image_path = image_base_dir / date / source_id / f"{xml_filename}.jpg"
    
    return image_path
