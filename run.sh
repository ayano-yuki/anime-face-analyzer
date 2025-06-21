#!/bin/bash

# アニメ顔画像解析システム実行スクリプト

echo "=================================="
echo "アニメ顔画像解析システム"
echo "=================================="

# 入力ディレクトリの確認
if [ ! -d "input" ]; then
    echo "inputディレクトリが見つかりません。作成します..."
    mkdir -p input
fi

# 入力画像の確認
image_count=$(find input -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)

if [ $image_count -eq 0 ]; then
    echo "警告: inputディレクトリに画像ファイルが見つかりません。"
    echo ""
    echo "サンプル画像を作成しますか？ (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "サンプル画像を作成中..."
        python3 setup_sample.py
        echo "サンプル画像を作成しました。"
    else
        echo "inputディレクトリにアニメ画像を配置してから再実行してください。"
        exit 1
    fi
else
    echo "入力画像: ${image_count}枚"
fi

# 出力ディレクトリの準備
if [ -d "output" ]; then
    echo "既存の出力ディレクトリをクリアしますか？ (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf output/*
        echo "出力ディレクトリをクリアしました。"
    fi
else
    mkdir -p output
fi

# modelsディレクトリの準備
mkdir -p models

echo ""
echo "Docker環境で実行を開始します..."
echo ""

# Docker Composeで実行
if command -v docker-compose &> /dev/null; then
    docker-compose up --build
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    docker compose up --build
else
    echo "エラー: DockerまたはDocker Composeが見つかりません。"
    echo "Dockerをインストールしてから再実行してください。"
    exit 1
fi

echo ""
echo "=================================="
echo "処理完了！"
echo "=================================="
echo "結果はoutputディレクトリに保存されました。"

# 結果の簡単な表示
if [ -f "output/similarity_stats.txt" ]; then
    echo ""
    echo "=== 結果サマリー ==="
    cat output/similarity_stats.txt
fi

echo ""
echo "詳細な結果を確認するには以下のファイルをご覧ください："
echo "- output/average_face.jpg (平均顔)"
echo "- output/similarity_stats.txt (統計情報)"
echo "- output/detailed_results.txt (詳細結果)"
