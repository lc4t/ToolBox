import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    DATETIME,
    DECIMAL,
    TIMESTAMP,
    BigInteger,
    Column,
    Date,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# 加载环境变量
load_dotenv()

# 获取数据库配置
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database configuration in .env file")

Base = declarative_base()


class TradingData(Base):
    __tablename__ = "trading_data"

    STORAGE_SCALE = 4  # 统一存储精度为4位小数

    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)
    # 价格字段统一使用4位小数存储
    open_price = Column(DECIMAL(20, STORAGE_SCALE), nullable=False)
    close_price = Column(DECIMAL(20, STORAGE_SCALE), nullable=False)
    high = Column(DECIMAL(20, STORAGE_SCALE), nullable=False)
    low = Column(DECIMAL(20, STORAGE_SCALE), nullable=False)
    volume = Column(BigInteger, nullable=False)
    create_time = Column(
        DATETIME, nullable=False, default=datetime.now, comment="创建时间"
    )
    update_time = Column(
        DATETIME,
        nullable=False,
        default=datetime.now,
        onupdate=datetime.now,
        comment="更新时间",
    )

    __table_args__ = ({"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},)


class SymbolInfo(Base):
    __tablename__ = "symbol_info"

    symbol = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False, comment="股票名称")
    listing_date = Column(Date, nullable=True, comment="上市时间")
    market = Column(String(20), nullable=False, comment="市场代码，如：SZ, SH, US")
    status = Column(
        String(20), nullable=False, default="active", comment="状态：active,delisted"
    )
    description = Column(String(500), nullable=True, comment="描述")
    create_time = Column(
        DATETIME, nullable=False, default=datetime.now, comment="创建时间"
    )
    update_time = Column(
        DATETIME,
        nullable=False,
        default=datetime.now,
        onupdate=datetime.now,
        comment="更新时间",
    )

    __table_args__ = ({"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},)


class DBClient:
    def __init__(
        self,
        host: str = DB_HOST,
        user: str = DB_USER,
        password: str = DB_PASSWORD,
        database: str = DB_NAME,
        port: int = DB_PORT,
    ) -> None:
        if not all([password, database]):
            raise ValueError("Database password and name are required")

        url = URL.create(
            drivername="mysql+pymysql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        self.engine = create_engine(url, echo=False, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)

    def init_db(self) -> None:
        """初始化数据库表"""
        Base.metadata.create_all(self.engine)

    def insert_trading_data(self, data: Dict[str, Any]) -> bool:
        """插入单条交易数据"""
        try:
            with self.Session() as session:
                # 统一使用4位小数存储
                price_fields = ["open_price", "close_price", "high", "low"]
                for field in price_fields:
                    if field in data:
                        data[field] = Decimal(str(data[field])).quantize(
                            Decimal(f"0.{'0' * TradingData.STORAGE_SCALE}")
                        )

                trading_data = TradingData(**data)
                session.add(trading_data)
                session.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error inserting data: {e}")
            return False

    def batch_insert_trading_data(self, data_list: List[Dict[str, Any]]) -> bool:
        """批量插入交易数据，忽略已存在的数据"""
        try:
            with self.Session() as session:
                # 处理价格精度
                processed_data = []
                for data in data_list:
                    processed_item = data.copy()
                    price_fields = ["open_price", "close_price", "high", "low"]

                    for field in price_fields:
                        if field in processed_item:
                            processed_item[field] = Decimal(
                                str(processed_item[field])
                            ).quantize(Decimal(f"0.{'0' * TradingData.STORAGE_SCALE}"))
                    processed_data.append(processed_item)

                # 使用 INSERT IGNORE 语法
                insert_stmt = insert(TradingData).prefix_with("IGNORE")
                session.execute(insert_stmt, processed_data)
                session.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error batch inserting data: {e}")
            return False

    def query_by_symbol_and_date_range(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> List[Dict[str, Any]]:
        """查询指定股票代码在日期范围内的数据"""
        try:
            with self.Session() as session:
                query = (
                    session.query(TradingData)
                    .filter(
                        TradingData.symbol == symbol,
                        TradingData.date.between(start_date, end_date),
                    )
                    .order_by(TradingData.date)
                )

                return [
                    {
                        "symbol": item.symbol,
                        "date": item.date,
                        "open_price": float(item.open_price),
                        "close_price": float(item.close_price),
                        "high": float(item.high),
                        "low": float(item.low),
                        "volume": item.volume,
                        "create_time": item.create_time,
                        "update_time": item.update_time,
                    }
                    for item in query.all()
                ]
        except SQLAlchemyError as e:
            logger.error(f"Error querying data: {e}")
            return []

    def insert_symbol_info(self, data: Dict[str, Any]) -> bool:
        """插入股票基本信息"""
        try:
            with self.Session() as session:
                symbol_info = SymbolInfo(**data)
                session.add(symbol_info)
                session.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error inserting symbol info: {e}")
            return False

    def batch_insert_symbol_info(self, data_list: List[Dict[str, Any]]) -> bool:
        """批量插入股票基本信息，忽略已存在的数据"""
        try:
            with self.Session() as session:
                # 使用 INSERT IGNORE 语法
                insert_stmt = insert(SymbolInfo).prefix_with("IGNORE")
                session.execute(insert_stmt, data_list)
                session.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error batch inserting symbol info: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取股票基本信息"""
        try:
            with self.Session() as session:
                info = (
                    session.query(SymbolInfo)
                    .filter(SymbolInfo.symbol == symbol)
                    .first()
                )

                if not info:
                    return None

                return {
                    "id": info.id,
                    "symbol": info.symbol,
                    "name": info.name,
                    "listing_date": info.listing_date,
                    "market": info.market,
                    "status": info.status,
                    "description": info.description,
                    "create_time": info.create_time,
                    "update_time": info.update_time,
                }
        except SQLAlchemyError as e:
            logger.error(f"Error querying symbol info: {e}")
            return None

    def update_symbol_info(self, symbol: str, data: Dict[str, Any]) -> bool:
        """更新股票基本信息"""
        try:
            with self.Session() as session:
                result = (
                    session.query(SymbolInfo)
                    .filter(SymbolInfo.symbol == symbol)
                    .update(data)
                )
                session.commit()
                return result > 0
        except SQLAlchemyError as e:
            logger.error(f"Error updating symbol info: {e}")
            return False

    def upsert_trading_data(self, data_list: List[Dict[str, Any]]) -> bool:
        """插入或更新交易数据（如果数据已存在且有变化则更新）"""
        try:
            with self.Session() as session:
                # 处理价格精度
                processed_data = []
                for data in data_list:
                    processed_item = data.copy()
                    price_fields = ["open_price", "close_price", "high", "low"]

                    for field in price_fields:
                        if field in processed_item:
                            processed_item[field] = Decimal(
                                str(processed_item[field])
                            ).quantize(Decimal(f"0.{'0' * TradingData.STORAGE_SCALE}"))
                    processed_data.append(processed_item)

                # 使用 INSERT ... ON DUPLICATE KEY UPDATE 语法
                insert_stmt = insert(TradingData).prefix_with("INSERT")
                update_dict = {
                    "open_price": insert_stmt.inserted.open_price,
                    "close_price": insert_stmt.inserted.close_price,
                    "high": insert_stmt.inserted.high,
                    "low": insert_stmt.inserted.low,
                    "volume": insert_stmt.inserted.volume,
                }
                upsert_stmt = insert_stmt.on_duplicate_key_update(update_dict)
                session.execute(upsert_stmt, processed_data)
                session.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error upserting data: {e}")
            return False
