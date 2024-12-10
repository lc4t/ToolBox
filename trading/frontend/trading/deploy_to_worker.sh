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
    yarn build
    echo "Starting Wrangler development server..."
    yarn wrangler pages dev out --port 8787
elif [ "$1" == "deploy" ]; then
    # 部署模式
    echo "Building frontend..."
    clean_build
    yarn build
    
    echo "Deploying to Cloudflare Pages..."
    yarn wrangler pages deploy out --no-cache
    
    echo "Deployment completed successfully!"
else
    echo "Usage: $0 [dev|deploy]"
    echo "  dev    - Start local development server"
    echo "  deploy - Deploy to Cloudflare Pages"
    exit 1
fi