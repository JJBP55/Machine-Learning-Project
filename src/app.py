from datetime import datetime, timedelta, timezone

from utils import (
    db_connect,
    init_db,
    get_binance_klines,
    save_candles_to_db,
    load_candles_from_db,
)

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS_OF_HISTORY = 1460                      #4 años


def main():
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_OF_HISTORY)

    print("Conectando a la base de datos...")
    engine = db_connect()
    init_db(engine)

    print(f"[1/3] Descargando {SYMBOL} ({INTERVAL}) desde {start_time.date()} hasta {end_time.date()}...")
    df_api = get_binance_klines(SYMBOL, INTERVAL, start_time, end_time)
    print(f"      -> {len(df_api)} velas recibidas de la API")

    print(f"[2/3] Guardando en Postgres...")
    inserted = save_candles_to_db(engine, df_api)
    print(f"      -> {inserted} filas nuevas insertadas")

    print(f"[3/3] Exportando CSV desde Postgres...")
    df_db = load_candles_from_db(engine, SYMBOL, INTERVAL)
    csv_path = f"data/raw/{SYMBOL}_{INTERVAL}.csv"
    df_db.to_csv(csv_path, index=False)
    print(f"      -> {len(df_db)} filas exportadas a {csv_path}")

    print("\nListo.")


if __name__ == "__main__":
    main()