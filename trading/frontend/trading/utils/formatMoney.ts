interface MarketConfig {
  currency: string;
  precision: number;
}

const MARKET_CONFIG: { [key: string]: MarketConfig } = {
  'HK': {
    currency: 'HK$',
    precision: 3,
  },
  'US': {
    currency: '$',
    precision: 2,
  },
  'CN': {
    currency: '¥',
    precision: 3,  // 中国市场通常显示3位小数
  }
};

export function getMarketFromSymbol(symbol: string): string {
  if (symbol.endsWith('.HK')) return 'HK';
  if (symbol.endsWith('.US')) return 'US';
  return 'CN';  // 默认中国市场
}

export function formatMoney(
  amount: number, 
  symbol: string,
): string {
  const market = getMarketFromSymbol(symbol);
  const config = MARKET_CONFIG[market];
  
  return `${config.currency}${amount.toFixed(config.precision)}`;
} 