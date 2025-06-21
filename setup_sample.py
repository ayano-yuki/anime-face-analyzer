#!/usr/bin/env python3
"""
サンプル画像をダウンロードするスクリプト
テスト用のアニメ画像を取得します
"""

import os
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_anime_faces():
    """テスト用のサンプルアニメ風画像を作成"""
    
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    
    # シンプルなアニメ風顔画像を生成
    def create_anime_face(size=(200, 200), face_color=(255, 220, 177), 
                         hair_color=(139, 69, 19), eye_color=(0, 100, 200)):
        """シンプルなアニメ風顔を描画"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # 髪（上部）
        hair_points = [
            (center_x - 60, center_y - 80),
            (center_x + 60, center_y - 80),
            (center_x + 50, center_y - 40),
            (center_x - 50, center_y - 40)
        ]
        draw.polygon(hair_points, fill=hair_color)
        
        # 顔（楕円）
        face_bbox = [center_x - 50, center_y - 40, center_x + 50, center_y + 60]
        draw.ellipse(face_bbox, fill=face_color, outline=(200, 180, 140), width=2)
        
        # 目
        # 左目
        left_eye_center = (center_x - 20, center_y - 10)
        draw.ellipse([left_eye_center[0] - 12, left_eye_center[1] - 8,
                     left_eye_center[0] + 12, left_eye_center[1] + 8], 
                    fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.ellipse([left_eye_center[0] - 6, left_eye_center[1] - 4,
                     left_eye_center[0] + 6, left_eye_center[1] + 4], 
                    fill=eye_color)
        draw.ellipse([left_eye_center[0] - 2, left_eye_center[1] - 2,
                     left_eye_center[0] + 2, left_eye_center[1] + 2], 
                    fill=(0, 0, 0))
        
        # 右目
        right_eye_center = (center_x + 20, center_y - 10)
        draw.ellipse([right_eye_center[0] - 12, right_eye_center[1] - 8,
                     right_eye_center[0] + 12, right_eye_center[1] + 8], 
                    fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.ellipse([right_eye_center[0] - 6, right_eye_center[1] - 4,
                     right_eye_center[0] + 6, right_eye_center[1] + 4], 
                    fill=eye_color)
        draw.ellipse([right_eye_center[0] - 2, right_eye_center[1] - 2,
                     right_eye_center[0] + 2, right_eye_center[1] + 2], 
                    fill=(0, 0, 0))
        
        # 鼻（小さな点）
        draw.ellipse([center_x - 2, center_y + 5, center_x + 2, center_y + 9], 
                    fill=(200, 150, 120))
        
        # 口
        mouth_points = [
            (center_x - 10, center_y + 25),
            (center_x, center_y + 30),
            (center_x + 10, center_y + 25)
        ]
        draw.polygon(mouth_points, fill=(200, 100, 100))
        
        return img
    
    # 異なる特徴を持つ複数のサンプル顔を作成
    samples = [
        {
            "filename": "sample_anime_1.png",
            "face_color": (255, 220, 177),
            "hair_color": (139, 69, 19),
            "eye_color": (0, 100, 200)
        },
        {
            "filename": "sample_anime_2.png", 
            "face_color": (255, 200, 160),
            "hair_color": (255, 215, 0),
            "eye_color": (0, 150, 0)
        },
        {
            "filename": "sample_anime_3.png",
            "face_color": (250, 210, 170),
            "hair_color": (75, 0, 130),
            "eye_color": (139, 0, 139)
        },
        {
            "filename": "sample_anime_4.png",
            "face_color": (255, 228, 196),
            "hair_color": (255, 20, 147),
            "eye_color": (255, 140, 0)
        },
        {
            "filename": "sample_anime_5.png",
            "face_color": (245, 215, 180),
            "hair_color": (0, 0, 0),
            "eye_color": (128, 0, 0)
        }
    ]
    
    print("サンプルアニメ画像を作成中...")
    
    for sample in samples:
        img = create_anime_face(
            face_color=sample["face_color"],
            hair_color=sample["hair_color"], 
            eye_color=sample["eye_color"]
        )
        
        filepath = os.path.join(input_dir, sample["filename"])
        img.save(filepath)
        print(f"作成完了: {filepath}")
    
    print(f"\n{len(samples)}個のサンプル画像を作成しました。")
    print(f"場所: {input_dir}/")
    print("\n実際のアニメ画像を使用する場合は、これらのファイルを置き換えてください。")

if __name__ == "__main__":
    create_sample_anime_faces()
