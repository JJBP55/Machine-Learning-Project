from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import pandas as pd
import requests
import time
from datetime import timezone

from models.candle_model import Base, Candle

# load the .env file variables
load_dotenv()

BINANCE_URL = "https://api.binance.us/api/v3/klines"

def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

def init_db(engine):
    #crea la tabla candles
    Base.metadata.create_all(engine)

#funciones

#descarga velas de binance entre start_time y end_time.
def get_binance_klines(symbol, interval, start_time, end_time):

    start_ms = int(start_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end_time.replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_candles = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        all_candles.extend(batch)
        cursor = batch[-1][6] + 1                                                       #close_time de la última vela + 1
        time.sleep(0.3)                                                                 #respetar el rate limit

        if len(batch) < 1000:
            break

    #convertir la respuesta de binance en un DataFrame limpio
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(all_candles, columns=columns)
    df = df.drop(columns=["ignore"])

    #convertir tipos, binance manda todo como string
    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    for col in float_cols:
        df[col] = df[col].astype(float)

    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["number_of_trades"] = df["number_of_trades"].astype("int64")

    df["symbol"] = symbol
    df["interval"] = interval

    df = df.drop_duplicates(subset=["symbol", "interval", "open_time"])
    return df.reset_index(drop=True)


#guarda las velas del DataFrame en la tabla candles usando SQLAlchemy
def save_candles_to_db(engine, df):

    session = Session(engine)
    inserted = 0

    for _, row in df.iterrows():
        #se verficia si ya existe (por symbol + interval + open_time)
        exists = session.execute(
            select(Candle).where(
                Candle.symbol == row["symbol"],
                Candle.interval == row["interval"],
                Candle.open_time == int(row["open_time"]),
            )
        ).first()

        if exists:
            continue

        candle = Candle(
            symbol=row["symbol"],
            interval=row["interval"],
            open_time=int(row["open_time"]),
            close_time=int(row["close_time"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            quote_asset_volume=row["quote_asset_volume"],
            number_of_trades=int(row["number_of_trades"]),
            taker_buy_base_volume=row["taker_buy_base_volume"],
            taker_buy_quote_volume=row["taker_buy_quote_volume"],
        )
        session.add(candle)
        inserted += 1

    session.commit()
    session.close()
    return inserted


#Lee todas las velas de un símbolo desde la base de datos,
#ordenadas cronológicamente, y las devuelve como DataFrame.
def load_candles_from_db(engine, symbol, interval="1h"):

    session = Session(engine)

    stmt = (
        select(Candle)
        .where(Candle.symbol == symbol, Candle.interval == interval)
        .order_by(Candle.open_time.asc())
    )
    candles = session.execute(stmt).scalars().all()
    session.close()

    #convertir la lista de objetos candle a DataFrame
    data = [{
        "open_time": c.open_time,
        "close_time": c.close_time,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "volume": c.volume,
        "quote_asset_volume": c.quote_asset_volume,
        "number_of_trades": c.number_of_trades,
        "taker_buy_base_volume": c.taker_buy_base_volume,
        "taker_buy_quote_volume": c.taker_buy_quote_volume,
        "symbol": c.symbol,
        "interval": c.interval,
    } for c in candles]

    return pd.DataFrame(data)    