# アニメ顔画像解析システム

アニメ画像からアニメの顔画像を抽出し、平均顔を作成して類似度を測定するシステムです。

## 機能

1. **アニメ顔検出・抽出**: アニメ画像からアニメキャラクターの顔を自動検出し、一定の解像度（128x128）に統一
2. **平均顔作成**: 抽出した全ての顔画像から平均顔を生成
3. **類似度測定**: 各顔画像と平均顔の類似度を複数の特徴量（ヒストグラム、LBP、HOG）を用いて計算

## 特徴

- **アニメ顔特化**: リアル顔検出ではなく、アニメキャラクター専用の検出モデル（lbpcascade_animeface）を使用
- **Docker対応**: 環境構築が簡単で、どこでも同じ結果を再現可能
- **詳細な結果出力**: 類似度統計、個別顔画像、ランキングなど豊富な出力

## 必要な環境

- Docker
- Docker Compose

## セットアップ・使用方法

### 1. プロジェクトのクローン/ダウンロード

```bash
# プロジェクトディレクトリに移動
cd anime-face-analyzer
```

### 2. 入力画像の準備

`input`ディレクトリにアニメ画像を配置してください。

```bash
# 入力画像の例
input/
├── anime_image1.jpg
├── anime_image2.png
├── anime_image3.jpg
└── ...
```

対応形式: JPG, JPEG, PNG, BMP, TIFF

### 3. Docker環境での実行

```bash
# Dockerイメージをビルドして実行
docker-compose up --build

# または、個別にビルド・実行
docker build -t anime-face-analyzer .
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output -v $(pwd)/models:/app/models anime-face-analyzer
```

### 4. 結果の確認

処理完了後、`output`ディレクトリに以下のファイルが生成されます：

```
output/
├── average_face.jpg                    # 平均顔画像
├── face_000_similarity_0.XXX.jpg      # 抽出された個別顔画像（類似度付き）
├── face_001_similarity_0.XXX.jpg
├── ...
├── similarity_stats.txt               # 類似度統計情報
└── detailed_results.txt               # 詳細結果（ランキング等）
```

## 出力ファイルの説明

### average_face.jpg
- 全ての抽出された顔画像から計算された平均顔

### face_XXX_similarity_Y.YYY.jpg
- 抽出された個別の顔画像
- ファイル名に類似度が含まれています

### similarity_stats.txt
- 平均類似度、最大・最小類似度、標準偏差などの統計情報

### detailed_results.txt
- 各顔の詳細情報（元画像名、顔番号、類似度）
- 類似度順ランキング

## 技術仕様

### アニメ顔検出
- **モデル**: lbpcascade_animeface.xml
- **手法**: Haar-like特徴量ベースのカスケード分類器
- **出力サイズ**: 128x128ピクセルに統一

### 特徴量抽出
1. **ヒストグラム特徴量**: グレースケール画像の輝度分布
2. **LBP (Local Binary Pattern)**: 局所的なテクスチャ特徴
3. **HOG (Histogram of Oriented Gradients)**: 勾配方向ヒストグラム

### 類似度計算
- **手法**: コサイン類似度
- **範囲**: 0-1（1が最も類似）

## トラブルシューティング

### 顔が検出されない場合
- 入力画像にアニメキャラクターの顔が明確に写っているか確認
- 画像の解像度が十分か確認（小さすぎる顔は検出されない場合があります）
- 画像の品質を確認（ぼやけていないか等）

### Docker関連のエラー
```bash
# Dockerイメージを再ビルド
docker-compose down
docker-compose up --build

# キャッシュをクリアして再ビルド
docker system prune -a
docker-compose up --build
```

### メモリ不足エラー
- 大量の画像を処理する場合、Dockerのメモリ制限を増やしてください
- または、入力画像を分割して処理してください

## カスタマイズ

### 顔画像のサイズ変更
`src/main.py`の`target_size`を変更：

```python
target_size = (256, 256)  # 256x256に変更
```

### 検出パラメータの調整
`src/anime_face_detector.py`の`detect_faces`メソッド内のパラメータを調整：

```python
faces = self.cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,    # より細かいスケール
    minNeighbors=3,      # より緩い検出
    minSize=(48, 48)     # より大きな最小サイズ
)
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 使用しているライブラリ・モデル

- OpenCV: 画像処理
- NumPy: 数値計算
- scikit-learn: 機械学習（類似度計算）
- lbpcascade_animeface: アニメ顔検出モデル（nagadomi氏作成）

## 参考文献

- [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) - アニメ顔検出モデル
