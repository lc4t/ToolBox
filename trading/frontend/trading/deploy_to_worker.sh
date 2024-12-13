#!/bin/bash

# 设置错误时退出
set -e

# 切换到前端目录
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# 确保在正确的目录
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found. Make sure you're in the correct directory."
    exit 1
fi

# 获取北京时间的时间戳（UTC+8）
get_beijing_time() {
    # 使用 TZ 环境变量设置时区为 Asia/Shanghai
    TZ='Asia/Shanghai' date '+%Y-%m-%dT%H:%M:%S+08:00'
}

# 清理函数
clean_build() {
    echo "Cleaning previous build..."
    rm -rf .next out
    rm -rf public/data/*.json
    find . -name ".wrangler" -type d -exec rm -rf {} +
}

if [ "$1" == "dev" ]; then
    # 本地开发模式
    echo "Starting local development server..."
    clean_build
    # 设置构建时间（北京时间）
    export NEXT_PUBLIC_BUILD_TIME=$(get_beijing_time)
    yarn build
    echo "Starting Wrangler development server..."
    yarn wrangler pages dev out --port 8787
elif [ "$1" == "deploy" ]; then
    # 部署模式
    echo "Building frontend..."
    clean_build
    # 设置构建时间（北京时间）
    export NEXT_PUBLIC_BUILD_TIME=$(get_beijing_time)
    yarn build
    
    echo "Deploying to Cloudflare Pages..."
    yarn wrangler pages deploy out --project-name trading
    
    echo "Deployment completed successfully!"
else
    echo "Usage: $0 [dev|deploy]"
    echo "  dev    - Start local development server"
    echo "  deploy - Deploy to Cloudflare Pages"
    exit 1
fi