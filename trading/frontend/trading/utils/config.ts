interface Config {
  googleAnalyticsId?: string;
  buildTime: string;
}

export const config: Config = {
  googleAnalyticsId: process.env.NEXT_PUBLIC_GA_ID,
  buildTime: process.env.NEXT_PUBLIC_BUILD_TIME || new Date().toISOString()
}; 