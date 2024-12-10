/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run `wrangler dev src/index.ts` in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run `wrangler deploy src/index.ts --name my-worker` to deploy your worker
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

import { getAssetFromKV } from '@cloudflare/kv-asset-handler';

interface Env {
	__STATIC_CONTENT: KVNamespace;
}

// 创建 Worker 实例
const worker = {
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext
	): Promise<Response> {
		try {
			// 尝试从 KV 存储获取静态资产
			return await getAssetFromKV({
				request,
				waitUntil: ctx.waitUntil.bind(ctx),
			});
		} catch {  // 移除未使用的参数
			// 如果资产不存在，返回 404
			return new Response('Not Found', { status: 404 });
		}
	},
};

export default worker;
