import { getAssetFromKV } from '@cloudflare/kv-asset-handler';

interface Env {
  __STATIC_CONTENT: KVNamespace;
}

const worker = {
  async fetch(request: Request): Promise<Response> {
    try {
      const url = new URL(request.url);
      let path = url.pathname;

      // 处理根路径
      if (path === '/' || path === '') {
        path = '/index.html';
      }

      // 尝试从静态文件目录读取文件
      const response = await fetch(request);
      if (response.ok) {
        // 设置缓存控制头
        const headers = new Headers(response.headers);
        headers.set('Cache-Control', 'no-cache');
        return new Response(response.body, {
          status: response.status,
          headers,
        });
      }

      // 如果文件不存在且不是 HTML 文件，返回 index.html（用于客户端路由）
      if (!path.endsWith('.html')) {
        const indexResponse = await fetch(new URL('/index.html', url.origin));
        if (indexResponse.ok) {
          return new Response(indexResponse.body, {
            status: 200,
            headers: {
              'Content-Type': 'text/html',
              'Cache-Control': 'no-cache',
            },
          });
        }
      }

      // 如果所有尝试都失败，返回 404
      return new Response('Not Found', {
        status: 404,
        headers: { 'Content-Type': 'text/plain' },
      });

    } catch (error) {
      console.error('Error serving:', error);
      return new Response('Internal Server Error', {
        status: 500,
        headers: { 'Content-Type': 'text/plain' },
      });
    }
  }
};

export default worker;