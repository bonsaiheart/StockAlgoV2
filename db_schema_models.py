from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint, JSON, \
    PrimaryKeyConstraint, func, TIMESTAMP, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Base = declarative_base()
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import MetaData

metadata = MetaData(schema='csvimport')
Base = declarative_base(metadata=metadata)

class Symbol(Base):
    __tablename__ = 'symbols'
    __table_args__ = ({'schema': 'csvimport'})
    symbol_name = Column(String,primary_key=True)
    description = Column(String(100))
    type = Column(String(10))  # Added field for type
class Option(Base):
    __tablename__ = 'options'
    # __table_args__ = (
    #     UniqueConstraint('underlying', 'expiration_date', 'strike', 'option_type'),
    # )
    __table_args__ = ({'schema': 'csvimport'})
    contract_id = Column(String,  primary_key=True)
    underlying = Column(String, ForeignKey('symbols.symbol_name', ondelete='CASCADE'))
    expiration_date = Column(Date)
    strike = Column(Float)
    option_type = Column(String)
    # Establish the relationship with Symbol
    symbol = relationship("Symbol", backref="options")

    contract_size = Column(Integer)
    description = Column(String)
    expiration_type = Column(String)
    # exch = Column(String)


class OptionQuote(Base):
    __tablename__ = 'option_quotes'
    __table_args__ = (
        UniqueConstraint('contract_id', 'fetch_timestamp', name='uq_option_quote_constraint'),{'schema': 'csvimport'}
    )

    contract_id = Column(String, ForeignKey('options.contract_id'), primary_key=True)
    fetch_timestamp = Column(TIMESTAMP(timezone=True), primary_key=True, server_default=func.now())

    option = relationship("Option", backref="quotes")
    root_symbol = Column(String)
    last = Column(Float)
    change = Column(Float)
    volume = Column(Integer)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    greeks = Column(JSON)
    change_percentage = Column(Float)
    # average_volume = Column(Integer)
    last_volume = Column(Integer)
    trade_date = Column(TIMESTAMP(timezone=True))
    prevclose = Column(Float)
    # week_52_high = Column(Float)
    # week_52_low = Column(Float)
    bidsize = Column(Integer)
    bidexch = Column(String)
    bid_date = Column(TIMESTAMP(timezone=True))
    asksize = Column(Integer)
    askexch = Column(String)
    ask_date = Column(TIMESTAMP(timezone=True))
    open_interest = Column(Integer)
    implied_volatility = Column(Float) #added this line
    realtime_calculated_greeks = Column(JSON)  # Added this line
    risk_free_rate = Column(Float)  # Added this line

class SymbolQuote(Base):
    __tablename__ = 'symbol_quotes'
    __table_args__ = (
        UniqueConstraint('symbol_name', 'fetch_timestamp', name='symbol_quote_unique_constraint'),{'schema': 'csvimport'}
    )
    symbol_name = Column(String, ForeignKey('symbols.symbol_name'), primary_key=True)
    fetch_timestamp = Column(TIMESTAMP(timezone=True), primary_key=True, server_default=func.now())

    symbol = relationship("Symbol", backref="symbol_quotes")
    last_trade_price = Column(Float)
    current_bid = Column(Float)
    current_ask = Column(Float)
    daily_open = Column(Float)
    daily_high = Column(Float)
    daily_low = Column(Float)
    previous_close = Column(Float)
    last_trade_volume = Column(Integer)
    daily_volume = Column(Integer)
    average_daily_volume = Column(Integer)
    last_trade_timestamp = Column(TIMESTAMP(timezone=True))
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    daily_change = Column(Float)
    daily_change_percentage = Column(Float)
    current_bidsize = Column(Integer)
    bidexch = Column(String)
    current_bid_date = Column(TIMESTAMP(timezone=True))
    current_asksize = Column(Integer)
    askexch = Column(String)
    current_ask_date = Column(TIMESTAMP(timezone=True))
    exch = Column(String(3))
    last_1min_timesale = Column(TIMESTAMP(timezone=True))
    last_1min_timestamp = Column(TIMESTAMP(timezone=True))
    last_1min_open = Column(Float)
    last_1min_high = Column(Float)
    last_1min_low = Column(Float)
    last_1min_close = Column(Float)
    last_1min_volume = Column(Integer)
    last_1min_vwap = Column(Float)

class TechnicalAnalysis(Base):
    __tablename__ = 'technical_analysis'
    ta_id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_name = Column(String, ForeignKey('symbols.symbol_name'), index=True)

    # 1mintimestamp_ = Column(DateTime, index=True, nullable=True)
    # timestamp_5min = Column(DateTime, index=True, nullable=True)
    # timestamp_15min = Column(DateTime, index=True, nullable=True)
    # Define other columns for each indicator and interval
    for interval in ["1min", "5min", "15min"]:
        # globals()[f"timestamp_{interval}"] = Column(DateTime, index=True)

        for indicator, data_type in [("timestamp", DateTime),
            ("price", Float),
            ("open", Float),
            ("high", Float),
            ("low", Float),
            ("close", Float),
            ("volume", Float),
            ("vwap", Float),
            ("SMA_20", Float),
            ("RSI_2", Float),
            ("RSI_7", Float),
            ("RSI_14", Float),
            ("RSI_21", Float),
            ("EMA_5", Float),
            ("EMA_14", Float),
            ("EMA_20", Float),
            ("EMA_50", Float),
            ("EMA_200", Float),
            ("MACD_12_26", Float),
            ("Signal_Line_12_26", Float),
            ("MACD_diff_12_26", Float),
            ("MACD_diff_prev_12_26", Float),
            ("MACD_signal_12_26", String),
            ("AwesomeOsc", Float),
            ("ADX", Float),
            ("CCI", Float),
            ("Williams_R", Float),
            ("PVO", Float),
            ("PPO", Float),
            ("CMF", Float),
            ("EoM", Float),
            ("OBV", Integer),
            ("MFI", Float),
            ("Keltner_Upper", Float),
            ("Keltner_Lower", Float),
            ("BB_high_20", Float),
            ("BB_mid_20", Float),
            ("BB_low_20", Float),
            ("VPT", Float),
        ]:
            column_name = f"{indicator}_{interval}"
            locals()[column_name] = Column(data_type, nullable=True)
        # Explicitly define an interval column after the loops2
    fetch_timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True, nullable=False)
    # interval = Column(String, index=True)
    __table_args__ = (
        PrimaryKeyConstraint('ta_id'),
        UniqueConstraint('symbol_name', 'fetch_timestamp', name='uq_symbol_interval_timestamps'),{'schema': 'csvimport'}
    )



class ProcessedOptionData(Base):
    __tablename__ = 'processed_option_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_name = Column(String, ForeignKey('symbols.symbol_name'), index=True)

    fetch_timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True, nullable=False)
    exp_date = Column(DateTime)
    current_stock_price = Column(Float)
    current_sp_change_lac = Column(Float)
    maximumpain = Column(Float)
    bonsai_ratio = Column(Float)
    bonsai_ratio_2 = Column(Float)
    b1_dividedby_b2 = Column(Float)
    b2_dividedby_b1 = Column(Float)
    pcr_vol = Column(Float)
    pcr_oi = Column(Float)
    # pcrv_cp_strike = Column(Float)
    # pcroi_cp_strike = Column(Float)
    pcrv_up1 = Column(Float)
    pcrv_up2 = Column(Float)
    pcrv_up3 = Column(Float)
    pcrv_up4 = Column(Float)
    pcrv_down1 = Column(Float)
    pcrv_down2 = Column(Float)
    pcrv_down3 = Column(Float)
    pcrv_down4 = Column(Float)
    pcroi_up1 = Column(Float)
    pcroi_up2 = Column(Float)
    pcroi_up3 = Column(Float)
    pcroi_up4 = Column(Float)
    pcroi_down1 = Column(Float)
    pcroi_down2 = Column(Float)
    pcroi_down3 = Column(Float)
    pcroi_down4 = Column(Float)
    itm_pcr_vol = Column(Float)
    itm_pcr_oi = Column(Float)
    itm_pcrv_up1 = Column(Float)
    itm_pcrv_up2 = Column(Float)
    itm_pcrv_up3 = Column(Float)
    itm_pcrv_up4 = Column(Float)
    itm_pcrv_down1 = Column(Float)
    itm_pcrv_down2 = Column(Float)
    itm_pcrv_down3 = Column(Float)
    itm_pcrv_down4 = Column(Float)
    itm_pcroi_up1 = Column(Float)
    itm_pcroi_up2 = Column(Float)
    itm_pcroi_up3 = Column(Float)
    itm_pcroi_up4 = Column(Float)
    itm_pcroi_down1 = Column(Float)
    itm_pcroi_down2 = Column(Float)
    itm_pcroi_down3 = Column(Float)
    itm_pcroi_down4 = Column(Float)
    itm_oi = Column(Integer)
    total_oi = Column(Integer)
    itm_contracts_percent = Column(Float)
    net_iv = Column(Float)
    net_itm_iv = Column(Float)
    net_iv_mp = Column(Float)
    # net_iv_lac = Column(Float)#TODO fix this
    niv_current_strike = Column(Float)
    niv_1higher_strike = Column(Float)
    niv_1lower_strike = Column(Float)
    niv_2higher_strike = Column(Float)
    niv_2lower_strike = Column(Float)
    niv_3higher_strike = Column(Float)
    niv_3lower_strike = Column(Float)
    niv_4higher_strike = Column(Float)
    niv_4lower_strike = Column(Float)
    niv_highers_minus_lowers1thru2 = Column(Float)
    niv_highers_minus_lowers1thru4 = Column(Float)
    niv_1thru2_avg_percent_from_mean = Column(Float)
    niv_1thru4_avg_percent_from_mean = Column(Float)
    niv_dividedby_oi = Column(Float)
    itm_avg_niv_dividedby_itm_oi = Column(Float)
    closest_strike_to_cp = Column(Float)


    __table_args__ = (
        UniqueConstraint('symbol_name', 'fetch_timestamp', name='uq_symbol_current_time_constraint'),  {'schema': 'csvimport'}

    )

class TimeSales(Base):
    __tablename__ = 'timesales'
    __table_args__ = {'schema': 'csvimport'}

    symbol = Column(String, primary_key=True)
    time = Column(DateTime, primary_key=True)
    timestamp = Column(DateTime)
    price = Column(Float)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    vwap = Column(Float)

    __table_args__ = (
        UniqueConstraint('symbol', 'time', name='timesales_pkey'),
    )