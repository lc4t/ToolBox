const fs = require('fs');
const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    if (!dev && isServer) {
      // 在生产构建时更新buildInfo.ts
      const buildTime = new Date().toISOString();
      const buildInfoPath = path.join(__dirname, 'trading/utils/buildInfo.ts');
      fs.writeFileSync(
        buildInfoPath,
        `// 这个文件会在构建时被更新\nexport const buildTime = '${buildTime}';\n`
      );
    }
    return config;
  },
};

module.exports = nextConfig; 