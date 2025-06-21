import cv2
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class FaceAnalyzer:
    """顔画像の平均顔作成と類似度測定を行うクラス"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_average_face(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        複数の顔画像から平均顔を作成
        
        Args:
            face_images: 顔画像のリスト
            
        Returns:
            平均顔画像
        """
        if not face_images:
            raise ValueError("顔画像が提供されていません")
        
        # 全ての画像を同じサイズに統一（既にリサイズ済みの前提）
        height, width = face_images[0].shape[:2]
        
        # 画像を浮動小数点型に変換
        face_arrays = []
        for face in face_images:
            if face.shape[:2] != (height, width):
                face = cv2.resize(face, (width, height))
            face_arrays.append(face.astype(np.float64))
        
        # 平均を計算
        average_face = np.mean(face_arrays, axis=0)
        
        # uint8に変換して返す
        return np.clip(average_face, 0, 255).astype(np.uint8)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        画像から特徴量を抽出
        
        Args:
            image: 入力画像
            
        Returns:
            特徴量ベクトル
        """
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # ヒストグラム特徴量
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # 正規化
        
        # LBP (Local Binary Pattern) 特徴量
        lbp = self._calculate_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = lbp_hist.flatten() / lbp_hist.sum()  # 正規化
        
        # HOG特徴量（簡易版）
        hog_features = self._calculate_simple_hog(gray)
        
        # 特徴量を結合
        features = np.concatenate([hist, lbp_hist, hog_features])
        
        return features
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        Local Binary Pattern を計算
        
        Args:
            image: グレースケール画像
            radius: 半径
            n_points: サンプリング点数
            
        Returns:
            LBP画像
        """
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                binary_string = ""
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if x >= 0 and x < height and y >= 0 and y < width:
                        if image[x, y] >= center:
                            binary_string += "1"
                        else:
                            binary_string += "0"
                    else:
                        binary_string += "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def _calculate_simple_hog(self, image: np.ndarray) -> np.ndarray:
        """
        簡易HOG特徴量を計算
        
        Args:
            image: グレースケール画像
            
        Returns:
            HOG特徴量
        """
        # Sobelフィルタで勾配を計算
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 勾配の大きさと方向を計算
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # 角度を0-180度に正規化
        angle = np.degrees(angle) % 180
        
        # ヒストグラムを作成（9つのビン）
        hist, _ = np.histogram(angle, bins=9, range=(0, 180), weights=magnitude)
        
        # 正規化
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist
    
    def calculate_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        2つの画像の類似度を計算
        
        Args:
            image1: 画像1
            image2: 画像2
            
        Returns:
            類似度（0-1の範囲、1が最も類似）
        """
        # 特徴量を抽出
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)
        
        # コサイン類似度を計算
        similarity = cosine_similarity([features1], [features2])[0][0]
        
        # 0-1の範囲に正規化
        return (similarity + 1) / 2
    
    def calculate_similarities_to_average(self, face_images: List[np.ndarray], 
                                        average_face: np.ndarray) -> List[float]:
        """
        各顔画像と平均顔の類似度を計算
        
        Args:
            face_images: 顔画像のリスト
            average_face: 平均顔画像
            
        Returns:
            類似度のリスト
        """
        similarities = []
        for face in face_images:
            similarity = self.calculate_similarity(face, average_face)
            similarities.append(similarity)
        
        return similarities
    
    def save_results(self, average_face: np.ndarray, face_images: List[np.ndarray], 
                    similarities: List[float], output_dir: str):
        """
        結果を保存
        
        Args:
            average_face: 平均顔画像
            face_images: 顔画像のリスト
            similarities: 類似度のリスト
            output_dir: 出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 平均顔を保存
        cv2.imwrite(os.path.join(output_dir, "average_face.jpg"), average_face)
        
        # 個別の顔画像を保存（類似度付き）
        for i, (face, similarity) in enumerate(zip(face_images, similarities)):
            filename = f"face_{i:03d}_similarity_{similarity:.3f}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), face)
        
        # 類似度の統計情報を保存
        stats = {
            "平均類似度": np.mean(similarities),
            "最大類似度": np.max(similarities),
            "最小類似度": np.min(similarities),
            "標準偏差": np.std(similarities)
        }
        
        with open(os.path.join(output_dir, "similarity_stats.txt"), "w", encoding="utf-8") as f:
            f.write("類似度統計情報\n")
            f.write("=" * 20 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n個別の類似度:\n")
            for i, similarity in enumerate(similarities):
                f.write(f"Face {i:03d}: {similarity:.4f}\n")
        
        print(f"結果を保存しました: {output_dir}")
        print(f"平均類似度: {stats['平均類似度']:.4f}")
