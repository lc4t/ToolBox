# ToolBox

工具箱

## Notion

### 1. auto_update_subscription.py

功能：筛选出订阅状态为“订阅中”的记录，如果其已经过期，就自动更新

使用流程：

1. 安装依赖：`python3.9 -m pip install -r requirements.txt`
2. 配置环境变量：`cp .env.example .env`，然后填入 Notion 的 Token 和数据库 ID
3. 运行脚本：`python3.9 auto_update_subscription.py`

注意事项：

1. Notion 的 Token 需要你自己创建`内部集成`
2. 需要在数据库右上角中，`连接自己的内部集成`
3. 复制数据库 ID
