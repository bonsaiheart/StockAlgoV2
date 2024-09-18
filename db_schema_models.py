from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint, JSON, \
    PrimaryKeyConstraint, func, TIMESTAMP, Date, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Base = declarative_base()
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import MetaData

metadata = MetaData(schema='csvimport')
Base = declarative_base(metadata=metadata)

class Symbol(Base):
    __tablename__ = 'symbols'
    __table_args__ = (
        Index('idx_symbols_symbol_name', 'symbol_name'),
        {'schema': 'csvimport'}
    )
    symbol_name = Column(String, primary_key=True)
    description = Column(String(100), nullable=True)
    type = Column(String(10), nullable=True)
    dividends = relationship("Dividend", back_populates="symbol")
class Option(Base):
    __tablename__ = 'options'
    __table_args__ = (
        Index('idx_options_underlying', 'underlying'),
        Index('idx_options_expiration_date', 'expiration_date'),
        Index('idx_options_strike', 'strike'),
        Index('idx_options_option_type', 'option_type'),
        {'schema': 'csvimport'}
    )
    contract_id = Column(String, primary_key=True)
    underlying = Column(String, ForeignKey('symbols.symbol_name', ondelete='CASCADE'))
    expiration_date = Column(Date)
    strike = Column(Float)
    option_type = Column(String)
    symbol = relationship("Symbol", backref="options")
    contract_size = Column(Integer, nullable=True)
    description = Column(String, nullable=True)
    expiration_type = Column(String, nullable=True)


class OptionQuote(Base):
    __tablename__ = 'option_quotes'
    # Indexes on contract_id, fetch_timestamp, and (contract_id, fetch_timestamp)
    __table_args__ = (
        UniqueConstraint('contract_id', 'fetch_timestamp', name='uq_option_quote_constraint'),

        Index('idx_option_quotes_contract_id', 'contract_id'),
        Index('idx_option_quotes_fetch_timestamp', 'fetch_timestamp'),
        Index('idx_option_quotes_contract_fetch', 'contract_id', 'fetch_timestamp'), {'schema': 'csvimport'},
    )

    contract_id = Column(String, ForeignKey('options.contract_id'), primary_key=True)
    fetch_timestamp = Column(TIMESTAMP(timezone=True), primary_key=True, server_default=func.now())

    option = relationship("Option", backref="quotes")
    root_symbol = Column(String, nullable=True)
    last = Column(Float, nullable=True)
    change = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    greeks = Column(JSON, nullable=True)
    change_percentage = Column(Float, nullable=True)
    last_volume = Column(Integer, nullable=True)
    trade_date = Column(TIMESTAMP(timezone=True), nullable=True)
    prevclose = Column(Float, nullable=True)
    bidsize = Column(Integer, nullable=True)
    bidexch = Column(String, nullable=True)
    bid_date = Column(TIMESTAMP(timezone=True), nullable=True)
    asksize = Column(Integer, nullable=True)
    askexch = Column(String, nullable=True)
    ask_date = Column(TIMESTAMP(timezone=True), nullable=True)
    open_interest = Column(Integer, nullable=True)
    implied_volatility = Column(Float, nullable=True)
    realtime_calculated_greeks = Column(JSON, nullable=True)
    risk_free_rate = Column(Float, nullable=True)


class SymbolQuote(Base):
    __tablename__ = 'symbol_quotes'
    __table_args__ = (
        UniqueConstraint('symbol_name', 'fetch_timestamp', name='symbol_quote_unique_constraint'),
        Index('idx_symbol_quotes_symbol_name', 'symbol_name'),
        Index('idx_symbol_quotes_fetch_timestamp', 'fetch_timestamp'),
        Index('idx_symbol_quotes_symbol_fetch', 'symbol_name', 'fetch_timestamp'),
        {'schema': 'csvimport'},
    )
    symbol_name = Column(String, ForeignKey('symbols.symbol_name'), primary_key=True)
    fetch_timestamp = Column(TIMESTAMP(timezone=True), primary_key=True, server_default=func.now())
    symbol = relationship("Symbol", backref="symbol_quotes")

    # Make all other fields nullable
    last_trade_price = Column(Float, nullable=True)
    current_bid = Column(Float, nullable=True)
    current_ask = Column(Float, nullable=True)
    daily_open = Column(Float, nullable=True)
    daily_high = Column(Float, nullable=True)
    daily_low = Column(Float, nullable=True)
    previous_close = Column(Float, nullable=True)
    last_trade_volume = Column(Integer, nullable=True)
    daily_volume = Column(Integer, nullable=True)
    average_daily_volume = Column(Integer, nullable=True)
    last_trade_timestamp = Column(TIMESTAMP(timezone=True), nullable=True)
    week_52_high = Column(Float, nullable=True)
    week_52_low = Column(Float, nullable=True)
    daily_change = Column(Float, nullable=True)
    daily_change_percentage = Column(Float, nullable=True)
    current_bidsize = Column(Integer, nullable=True)
    bidexch = Column(String, nullable=True)
    current_bid_date = Column(TIMESTAMP(timezone=True), nullable=True)
    current_asksize = Column(Integer, nullable=True)
    askexch = Column(String, nullable=True)
    current_ask_date = Column(TIMESTAMP(timezone=True), nullable=True)
    exch = Column(String(3), nullable=True)
    last_1min_timesale = Column(TIMESTAMP(timezone=True), nullable=True)
    last_1min_timestamp = Column(TIMESTAMP(timezone=True), nullable=True)
    last_1min_open = Column(Float, nullable=True)
    last_1min_high = Column(Float, nullable=True)
    last_1min_low = Column(Float, nullable=True)
    last_1min_close = Column(Float, nullable=True)
    last_1min_volume = Column(Integer, nullable=True)
    last_1min_vwap = Column(Float, nullable=True)
class Dividend(Base):
    __tablename__ = 'dividends'
    __table_args__ = (
        Index('idx_dividends_symbol_name', 'symbol_name'),
        Index('idx_dividends_ex_date', 'ex_date'),
        {'schema': 'csvimport'}
    )
    id = Column(Integer, primary_key=True)
    symbol_name = Column(String, ForeignKey('symbols.symbol_name'))
    dividend_type = Column(String, nullable=True)
    ex_date = Column(Date, nullable=True)
    cash_amount = Column(Float, nullable=True)
    currency_id = Column(String, nullable=True)
    declaration_date = Column(Date, nullable=True)
    frequency = Column(Integer, nullable=True)
    pay_date = Column(Date, nullable=True)
    record_date = Column(Date, nullable=True)
    symbol = relationship("Symbol", back_populates="dividends")


class TechnicalAnalysis(Base):
    __tablename__ = 'technical_analysis'
    __table_args__ = (
        UniqueConstraint('symbol_name', 'interval', 'timestamp', name='uq_ta_symbol_interval_timestamp'),
        Index('idx_ta_symbol_name', 'symbol_name'),
        Index('idx_ta_timestamp', 'timestamp'),
        Index('idx_ta_fetch_timestamp', 'fetch_timestamp'),
        {'schema': 'csvimport'}
    )

    id = Column(Integer, primary_key=True)
    symbol_name = Column(String, ForeignKey('csvimport.symbols.symbol_name'), nullable=False)
    interval = Column(String, nullable=False)  # '1min', '5min', '15min'
    timestamp = Column(DateTime(timezone=True), nullable=False)
    fetch_timestamp = Column(DateTime(timezone=True), nullable=False)

    # OHLCV data
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    vwap = Column(Float)

    # Technical indicators
    ema_5 = Column(Float)
    ema_14 = Column(Float)
    ema_20 = Column(Float)
    ema_50 = Column(Float)
    ema_200 = Column(Float)

    rsi_2 = Column(Float)
    rsi_7 = Column(Float)
    rsi_14 = Column(Float)
    rsi_21 = Column(Float)

    awesome_oscillator = Column(Float)
    sma_20 = Column(Float)
    adx = Column(Float)
    cci = Column(Float)
    cmf = Column(Float)
    eom = Column(Float)
    obv = Column(Float)
    mfi = Column(Float)

    macd = Column(Float)
    macd_signal = Column(Float)
    macd_diff = Column(Float)
    macd_diff_prev = Column(Float)

    williams_r = Column(Float)
    pvo = Column(Float)
    ppo = Column(Float)
    keltner_upper = Column(Float)
    keltner_lower = Column(Float)
    bb_high = Column(Float)
    bb_mid = Column(Float)
    bb_low = Column(Float)
    vpt = Column(Float)

    symbol = relationship("Symbol", back_populates="technical_analysis")


# Add this line to the Symbol class
Symbol.technical_analysis = relationship("TechnicalAnalysis", back_populates="symbol")


class OptimizedProcessedOptionData(Base):
    __tablename__ = 'optimized_processed_option_data'
    __table_args__ = (
        UniqueConstraint('symbol_name', 'fetch_timestamp', 'exp_date', name='uq_symbol_timestamp_expiry'),
        {'schema': 'csvimport'}
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_name = Column(String, ForeignKey('symbols.symbol_name'), index=True)
    fetch_timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True, nullable=False)
    exp_date = Column(DateTime, index=True)

    # Market data
    current_stock_price = Column(Float)
    current_sp_change_lac = Column(Float)

    # Key metrics
    max_pain = Column(Float)
    bonsai_ratio = Column(Float)
    pcr_vol = Column(Float)
    pcr_oi = Column(Float)
    itm_pcr_vol = Column(Float)
    itm_pcr_oi = Column(Float)

    # Open Interest
    itm_oi = Column(Integer)
    total_oi = Column(Integer)
    itm_contracts_percent = Column(Float)

    # IV metrics
    avg_call_iv = Column(Float)
    avg_put_iv = Column(Float)
    net_iv = Column(Float)
    net_itm_iv = Column(Float)

    # Greeks averages
    avg_delta = Column(Float)
    avg_gamma = Column(Float)
    avg_theta = Column(Float)
    avg_vega = Column(Float)

    # Volume metrics
    total_volume = Column(Integer)
    vwap = Column(Float)

    # Additional useful metrics
    closest_strike_to_cp = Column(Float)
    atm_iv = Column(Float)  # At-the-money implied volatility

    # JSON column for less frequently used or derived metrics
    additional_metrics = Column(JSON)
class ProcessedOptionData(Base):
    __tablename__ = 'processed_option_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_name = Column(String, ForeignKey('symbols.symbol_name'), index=True)

    fetch_timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True, nullable=False)
    exp_date = Column((DateTime), index=True)
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

# class TimeSales(Base):
#     __tablename__ = 'timesales'
#     # Indexes on symbol, time, and (symbol, time)
#     __table_args__ = (
#         UniqueConstraint('symbol', 'time', name='timesales_pkey'),
#
#         Index('idx_timesales_symbol', 'symbol'),
#         Index('idx_timesales_time', 'time'),
#         Index('idx_timesales_symbol_time', 'symbol', 'time'), {'schema': 'csvimport'}
#     )
#
#
#     symbol = Column(String, primary_key=True)
#     time = Column(DateTime, primary_key=True)
#     timestamp = Column(DateTime)
#     price = Column(Float)
#     open = Column(Float)
#     high = Column(Float)
#     low = Column(Float)
#     close = Column(Float)
#     volume = Column(Integer)
#     vwap = Column(Float)
#


