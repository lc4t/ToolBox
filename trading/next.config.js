const fs = require('fs');
const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    if (!dev && isServer) {
      const buildTime = new Date().toISOString();
      const gaId = process.env.NEXT_PUBLIC_GA_ID;
      
      // 更新配置文件
      const configPath = path.join(__dirname, 'trading/utils/config.ts');
      fs.writeFileSync(
        configPath,
        `// 这个文件会在构建时被更新
interface Config {
  googleAnalyticsId?: string;
  buildTime: string;
}

export const config: Config = {
  googleAnalyticsId: ${gaId ? `"${gaId}"` : 'undefined'},
  buildTime: "${buildTime}"
};\n`
      );
    }
    return config;
  },
};

module.exports = nextConfig; 