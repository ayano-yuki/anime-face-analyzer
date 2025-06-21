#!/usr/bin/env python3
"""
アニメ顔画像解析メインスクリプト

機能:
1. アニメ画像からアニメの顔画像を切り抜き、全ての顔画像を一定の解像度に揃える
2. アニメ顔画像の平均顔を作成する
3. 作成した平均顔と抽出した顔画像の類似度を測定する
"""

import os
import sys
import glob
from typing import List
import cv2
import numpy as np
from tqdm import tqdm

# 相対インポートのためのパス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anime_face_detector import AnimeFaceDetector
from face_analyzer import FaceAnalyzer

def main():
    """メイン処理"""
    print("=" * 50)
    print("アニメ顔画像解析システム")
    print("=" * 50)
    
    # 設定
    input_dir = "input"
    output_dir = "output"
    target_size = (128, 128)  # 顔画像の統一サイズ
    
    # ディレクトリの存在確認
    if not os.path.exists(input_dir):
        print(f"エラー: 入力ディレクトリが見つかりません: {input_dir}")
        print("inputディレクトリにアニメ画像を配置してください。")
        return
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 対応する画像形式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 入力画像ファイルを取得
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"エラー: {input_dir}ディレクトリに画像ファイルが見つかりません")
        print("対応形式: jpg, jpeg, png, bmp, tiff")
        return
    
    print(f"処理対象画像: {len(image_files)}枚")
    
    # アニメ顔検出器を初期化
    print("\n1. アニメ顔検出器を初期化中...")
    try:
        detector = AnimeFaceDetector()
        print("✓ アニメ顔検出器の初期化完了")
    except Exception as e:
        print(f"エラー: アニメ顔検出器の初期化に失敗しました: {e}")
        return
    
    # 顔解析器を初期化
    analyzer = FaceAnalyzer()
    
    # 全ての画像から顔を抽出
    print("\n2. 画像から顔を抽出中...")
    all_faces = []
    face_info = []  # (画像ファイル名, 顔のインデックス) の情報
    
    for image_file in tqdm(image_files, desc="顔抽出"):
        try:
            faces = detector.process_image(image_file, target_size)
            for i, face in enumerate(faces):
                all_faces.append(face)
                face_info.append((os.path.basename(image_file), i))
        except Exception as e:
            print(f"警告: {image_file}の処理中にエラーが発生しました: {e}")
            continue
    
    if not all_faces:
        print("エラー: 顔が検出されませんでした")
        print("入力画像にアニメキャラクターの顔が含まれているか確認してください")
        return
    
    print(f"✓ 合計 {len(all_faces)} 個の顔を抽出しました")
    
    # 平均顔を作成
    print("\n3. 平均顔を作成中...")
    try:
        average_face = analyzer.create_average_face(all_faces)
        print("✓ 平均顔の作成完了")
    except Exception as e:
        print(f"エラー: 平均顔の作成に失敗しました: {e}")
        return
    
    # 類似度を計算
    print("\n4. 類似度を計算中...")
    try:
        similarities = analyzer.calculate_similarities_to_average(all_faces, average_face)
        print("✓ 類似度計算完了")
    except Exception as e:
        print(f"エラー: 類似度計算に失敗しました: {e}")
        return
    
    # 結果を保存
    print("\n5. 結果を保存中...")
    try:
        analyzer.save_results(average_face, all_faces, similarities, output_dir)
        
        # 詳細情報も保存
        save_detailed_info(face_info, similarities, output_dir)
        
        print("✓ 結果の保存完了")
    except Exception as e:
        print(f"エラー: 結果の保存に失敗しました: {e}")
        return
    
    # 統計情報を表示
    print("\n" + "=" * 50)
    print("処理完了！")
    print("=" * 50)
    print(f"処理した画像数: {len(image_files)}")
    print(f"抽出した顔数: {len(all_faces)}")
    print(f"平均類似度: {np.mean(similarities):.4f}")
    print(f"最大類似度: {np.max(similarities):.4f}")
    print(f"最小類似度: {np.min(similarities):.4f}")
    print(f"標準偏差: {np.std(similarities):.4f}")
    print(f"\n結果は {output_dir} ディレクトリに保存されました")
    
    # 最も類似度の高い顔と低い顔を表示
    max_idx = np.argmax(similarities)
    min_idx = np.argmin(similarities)
    
    print(f"\n最も平均顔に近い顔:")
    print(f"  ファイル: {face_info[max_idx][0]}, 顔番号: {face_info[max_idx][1]}")
    print(f"  類似度: {similarities[max_idx]:.4f}")
    
    print(f"\n最も平均顔から遠い顔:")
    print(f"  ファイル: {face_info[min_idx][0]}, 顔番号: {face_info[min_idx][1]}")
    print(f"  類似度: {similarities[min_idx]:.4f}")

def save_detailed_info(face_info: List[tuple], similarities: List[float], output_dir: str):
    """詳細情報を保存"""
    with open(os.path.join(output_dir, "detailed_results.txt"), "w", encoding="utf-8") as f:
        f.write("詳細結果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("各顔の詳細情報:\n")
        f.write("-" * 30 + "\n")
        
        for i, ((filename, face_idx), similarity) in enumerate(zip(face_info, similarities)):
            f.write(f"Face {i:03d}:\n")
            f.write(f"  元画像: {filename}\n")
            f.write(f"  顔番号: {face_idx}\n")
            f.write(f"  類似度: {similarity:.4f}\n")
            f.write(f"  保存名: face_{i:03d}_similarity_{similarity:.3f}.jpg\n")
            f.write("\n")
        
        # 類似度でソートした結果も追加
        sorted_indices = np.argsort(similarities)[::-1]  # 降順
        
        f.write("\n類似度順ランキング:\n")
        f.write("-" * 30 + "\n")
        
        for rank, idx in enumerate(sorted_indices, 1):
            filename, face_idx = face_info[idx]
            similarity = similarities[idx]
            f.write(f"{rank:2d}位: {filename} (顔{face_idx}) - 類似度: {similarity:.4f}\n")

if __name__ == "__main__":
    main()
