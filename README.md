
## 使用方法

你可以通过命令行参数来控制脚本的行为。以下是可用的命令行参数：

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
- `--download-webhook-url`：指定下载 webhook URL，启用下载模式。
- `--download-filter-status`：指定下载过滤状态，默认为 "download_filter"。
- `--download-wait-status`：指定下载等待状态，默认为 "download_wait"。

## 示例命令

1. 抓取最旧的 5 个 fetch，并获取最新的视频：

    ```bash
    python bili2cubox.py --fetch-count 5
    ```

2. 获取 fetcher 的所有视频，而不是最新的 7 天：

    ```bash
    python bili2cubox.py --all-videos
    ```

3. 设置过滤中的天数限制为 10 天：

    ```bash
    python bili2cubox.py --days-limit 10
    ```

4. 更新 webhook_url 为指定值的 fetcher：

    ```bash
    python bili2cubox.py --webhook-url your_webhook_url
    ```

5. 排除 webhook_url 为指定值的 fetcher：

    ```bash
    python bili2cubox.py --exclude-webhook-url your_webhook_url
    ```

6. 初始化或更新数据库模式：

    ```bash
    python bili2cubox.py --init-db
    ```

7. 重定向 webhook URL 的 host：

    ```bash
    python bili2cubox.py --redirect-host new_host
    ```

8. 使用黑名单和白名单标签：

    ```bash
    python bili2cubox.py --black-tag "不感兴趣,广告" --white-tag "重要,收藏"
    ```

9. 启用下载模式并指定下载状态：

    ```bash
    python bili2cubox.py --download-webhook-url your_download_webhook_url --download-filter-status "video_filter" --download-wait-status "video_pass"
    ```

## 注意事项

1. 确保在运行脚本之前已正确配置 `.env` 文件。
2. 使用 `--init-db` 参数初始化数据库后，才能正常使用其他功能。
3. 下载模式通过指定 `--download-webhook-url` 来启用，无需单独的 `--download-flag` 参数。