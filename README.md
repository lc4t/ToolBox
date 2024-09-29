# ToolBox

工具箱

## 环境配置

在开始使用之前，请确保你已经配置好环境变量。你可以在项目根目录下创建一个 `.env` 文件，并根据以下示例进行配置：

### `.env` 文件示例

```plaintext:bilibili/.env
# 数据库连接字符串
DB_CONNECTION_STRING=mysql+pymysql://[USER]:[PASSWORD]@[HOST]:[PORT]/[DBNAME]{?charset=utf8mb4}

# Cubox API URL
CUBOX_API_URL=https://cubox.pro/c/api/save/[YOUR_API_PATH]

# Omnivore Authorization
OMNIVORE_AUTHORIZATION=[YOUR_OMNIVORE_AUTHORIZATION]

# 可选的 Feed Token
FEED_TOKEN=[YOUR_FEED_TOKEN]  # 如果没有 token，可以留空
```

### `.env.example` 文件示例

你也可以参考项目中的 `.env.example` 文件：

```plaintext:notion/.env.example
NOTION_TOKEN=
SUBSCRIPTION_DATABASE_ID=
DB_CONNECTION_STRING=your_database_connection_string
CUBOX_API_URL=your_cubox_api_url
OMNIVORE_AUTHORIZATION=your_omnivore_authorization
FEED_TOKEN=your_feed_token  # 如果没有 token，可以留空
```

## 安装依赖

在项目根目录下运行以下命令来安装所需的依赖：

```bash
pip install -r bilibili/requirements.txt
```

## Bili2Cubox

Bili2Cubox 是一个用于抓取 Bilibili 视频并推送到 Cubox 和 IFTTT 的脚本。该脚本支持通过命令行参数配置抓取和推送行为。

### 使用方法

你可以通过命令行参数来控制脚本的行为。以下是一些常用的命令行参数：

- `--filter`：过滤视频的关键词。
- `--fetch-count`：指定要抓取的最旧的 fetch 数量，默认为 3。
- `--all-videos`：如果设置此参数，将获取 fetcher 的所有视频，而不是最新的 7 天。
- `--days-limit`：设置过滤中的天数限制，默认为 7 天。
- `--webhook-url`：指定要更新的 fetcher 的 webhook_url。
- `--exclude-webhook-url`：指定要排除的 fetcher 的 webhook_url。
- `--init-db`：初始化或更新数据库模式。
- `--redirect-host`：重定向 webhook URL 的 host。
- `--black-tag`：指定黑名单标签，命中该标签时将状态设置为 filter。
- `--white-tag`：指定白名单标签，命中该标签时将状态设置为 new。
- `--download-flag`：启用下载标志。
- `--download-filter-status`：指定下载过滤状态，默认为 "download_filter"。
- `--download-wait-status`：指定下载等待状态，默认为 "download_wait"。
- `--download-webhook-url`：指定下载 webhook URL。

### 示例命令

1. 抓取最旧的 5 个 fetch，并获取最新的视频：

    ```bash
    python bilibili/bili2cubox.py --fetch-count 5
    ```

2. 获取 fetcher 的所有视频，而不是最新的 7 天：

    ```bash
    python bilibili/bili2cubox.py --all-videos
    ```

3. 设置过滤中的天数限制为 10 天：

    ```bash
    python bilibili/bili2cubox.py --days-limit 10
    ```

4. 更新 webhook_url 为指定值的 fetcher：

    ```bash
    python bilibili/bili2cubox.py --webhook-url your_webhook_url
    ```

5. 排除 webhook_url 为指定值的 fetcher：

    ```bash
    python bilibili/bili2cubox.py --exclude-webhook-url your_webhook_url
    ```

6. 初始化或更新数据库模式：

    ```bash
    python bilibili/bili2cubox.py --init-db
    ```

7. 重定向 webhook URL 的 host：

    ```bash
    python bilibili/bili2cubox.py --redirect-host new_host
    ```

8. 使用黑名单和白名单标签：

    ```bash
    python bilibili/bili2cubox.py --black-tag "不感兴趣,广告" --white-tag "重要,收藏"
    ```

9. 启用下载标志并指定下载状态：

    ```bash
    python bilibili/bili2cubox.py --download-flag --download-filter-status "video_filter" --download-wait-status "video_pass"
    ```

10. 指定下载 webhook URL：

    ```bash
    python bilibili/bili2cubox.py --download-webhook-url your_download_webhook_url
    ```

## Notion

### 1. auto_update_subscription.py

功能：筛选出订阅状态为"订阅中"的记录，如果其已经过期，就自动更新

使用流程：

1. 安装依赖：`python3.9 -m pip install -r requirements.txt`
2. 配置环境变量：`cp .env.example .env`，然后填入 Notion 的 Token 和数据库 ID
3. 运行脚本：`python3.9 auto_update_subscription.py`

注意事项：

1. Notion 的 Token 需要你自己创建`内部集成`
2. 需要在数据库右上角中，`连接自己的内部集成`
3. 复制数据库 ID

