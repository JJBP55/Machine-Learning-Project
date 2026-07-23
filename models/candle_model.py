from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, BigInteger, UniqueConstraint


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    interval: Mapped[str] = mapped_column(String(10), index=True)
    open_time: Mapped[int] = mapped_column(BigInteger, index=True)
    close_time: Mapped[int] = mapped_column(BigInteger)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    quote_asset_volume: Mapped[float] = mapped_column(Float)
    number_of_trades: Mapped[int] = mapped_column(BigInteger)
    taker_buy_base_volume: Mapped[float] = mapped_column(Float)
    taker_buy_quote_volume: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_symbol_interval_time"),
    )