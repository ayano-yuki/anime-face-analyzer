import cv2
import numpy as np
import os
import requests
from typing import List, Tuple, Optional

class AnimeFaceDetector:
    """アニメ顔検出用のクラス"""
    
    def __init__(self, model_path: str = "models/lbpcascade_animeface.xml"):
        self.model_path = model_path
        self.cascade = None
        self._download_model_if_needed()
        self._load_model()
    
    def _download_model_if_needed(self):
        """アニメ顔検出モデルをダウンロード"""
        if not os.path.exists(self.model_path):
            print("アニメ顔検出モデルをダウンロード中...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # lbpcascade_animeface.xmlをダウンロード
            url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(self.model_path, 'wb') as f:
                    f.write(response.content)
                print(f"モデルを保存しました: {self.model_path}")
            else:
                raise Exception(f"モデルのダウンロードに失敗しました: {response.status_code}")
    
    def _load_model(self):
        """モデルを読み込み"""
        if os.path.exists(self.model_path):
            self.cascade = cv2.CascadeClassifier(self.model_path)
            if self.cascade.empty():
                raise Exception(f"モデルの読み込みに失敗しました: {self.model_path}")
        else:
            raise Exception(f"モデルファイルが見つかりません: {self.model_path}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        画像からアニメ顔を検出
        
        Args:
            image: 入力画像 (BGR形式)
            
        Returns:
            検出された顔の座標リスト [(x, y, w, h), ...]
        """
        if self.cascade is None:
            raise Exception("モデルが読み込まれていません")
        
        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 顔検出
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        
        return faces.tolist()
    
    def extract_face(self, image: np.ndarray, face_coords: Tuple[int, int, int, int], 
                    target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        画像から顔部分を切り抜いてリサイズ
        
        Args:
            image: 入力画像
            face_coords: 顔の座標 (x, y, w, h)
            target_size: 出力サイズ (width, height)
            
        Returns:
            切り抜かれた顔画像
        """
        x, y, w, h = face_coords
        
        # 顔部分を切り抜き
        face_image = image[y:y+h, x:x+w]
        
        # 指定サイズにリサイズ
        resized_face = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return resized_face
    
    def process_image(self, image_path: str, target_size: Tuple[int, int] = (128, 128)) -> List[np.ndarray]:
        """
        画像ファイルを処理してアニメ顔を抽出
        
        Args:
            image_path: 画像ファイルのパス
            target_size: 出力サイズ
            
        Returns:
            抽出された顔画像のリスト
        """
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"画像の読み込みに失敗しました: {image_path}")
        
        # 顔検出
        faces = self.detect_faces(image)
        
        # 顔を切り抜き
        extracted_faces = []
        for face_coords in faces:
            face_image = self.extract_face(image, face_coords, target_size)
            extracted_faces.append(face_image)
        
        return extracted_faces
