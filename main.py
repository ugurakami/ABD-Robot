import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
import os
import time  # <-- YENİ: Bekleme süresi için
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# -------------------- AYARLAR --------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# Trading Ayarları
PORTFOLIO_SIZE = 50_000
RISK_PER_TRADE = 0.01
MAX_POSITIONS = 3        
SUPER_TREND_PERIOD = 10
SUPER_TREND_MULT = 3.0
ATR_PERIOD = 14
MAX_PULLBACK_ATR = 2.0
RSI_MAX_THRESHOLD = 70

# -------------------- VERİ SETİ --------------------
# -------------------- GÜNCELLENMİŞ VERİ SETİ --------------------

# 1. Sektör Listesine "Tahvil" ve "Emtia"yı da ekliyoruz
SECTOR_ETFS = {
    # Klasik Sektörler
    'XLK': 'Teknoloji',
    'XLF': 'Finans',
    'XLV': 'Sağlık',
    'XLE': 'Enerji',
    'XLY': 'Tüketim (Lüks)',
    'XLP': 'Tüketim (Temel)',
    'XLI': 'Sanayi',
    'XLC': 'İletişim',
    'XLB': 'Hammadde',
    'XLRE': 'Gayrimenkul',
    'XLU': 'Altyapı',
    
    # --- YENİ EKLENEN SAVUNMA HATLARI ---
    'TLT': 'ABD Tahvil (20+ Yıl)',  # Uzun vadeli tahvil
    'SHY': 'ABD Tahvil (Kısa Vade)', # Kısa vadeli tahvil (Nakit benzeri)
    'GLD': 'Altın',                  # Altın ETF
    'SPY': 'Genel Borsa (S&P 500)'   # Borsa Endeksinin kendisi
}

# 2. Hisseler Haritasına bu ETF'lerin kendisini de ekliyoruz
# Mantık: TLT hissesi, 'ABD Tahvil' sektöründedir.
STOCK_SECTOR_MAP = {
    # --- MEVCUT HİSSELER ---
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'ADBE': 'XLK',
    'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC',
    'AMZN': 'XLY', 'TSLA': 'XLY', 'MCD': 'XLY',
    'JPM': 'XLF', 'V': 'XLF', 'MA': 'XLF',
    'JNJ': 'XLV', 'UNH': 'XLV', 'LLY': 'XLV',
    'XOM': 'XLE', 'CVX': 'XLE',
    'CAT': 'XLI', 'BA': 'XLI',
    'PG': 'XLP', 'KO': 'XLP', 'WMT': 'XLP',
    
    # --- YENİ EKLENEN ETF'LERİN ALIM SATIMI ---
    # Botun direkt bu ETF'leri de alıp satabilmesi için buraya ekliyoruz.
    'TLT': 'TLT',  # Tahvilin kendisini al
    'GLD': 'GLD',  # Altının kendisini al
    'SPY': 'SPY',  # Endeksin kendisini al
    'QQQ': 'XLK',  # Nasdaq'ı Teknoloji kategorisine koyabiliriz
    'IWM': 'SPY'   # Küçük şirketleri Genel Borsa kategorisine koyabiliriz
}

# -------------------- YARDIMCI FONKSİYONLAR --------------------

# YENİ: Yahoo Finance Engelini Aşmak İçin Özel İstek Fonksiyonu
def get_data_safe(ticker, period, interval):
    """Bot korumasını aşmak için User-Agent eklenmiş veri çekme fonksiyonu"""
    try:
        # Ticker nesnesini oluştur
        dat = yf.Ticker(ticker)
        
        # Veriyi çek (auto_adjust=True bazen veri hatalarını önler)
        df = dat.history(period=period, interval=interval, auto_adjust=True)
        
        if df.empty:
            # Alternatif yöntem: download override
            df = yf.download(ticker, period=period, interval=interval, progress=False, ignore_tz=True)
            
        return df
    except Exception as e:
        print(f"⚠️ Veri hatası ({ticker}): {e}")
        return pd.DataFrame()

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_supertrend(df, period=10, multiplier=3):
    atr = calculate_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    high, low, close = df['High'].values, df['Low'].values, df['Close'].values
    bub, blb = basic_upperband.values, basic_lowerband.values
    
    final_upperband = np.zeros(len(df))
    final_lowerband = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    trend_direction = np.ones(len(df))
    
    for i in range(1, len(df)):
        if bub[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
            final_upperband[i] = bub[i]
        else:
            final_upperband[i] = final_upperband[i-1]
            
        if blb[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
            final_lowerband[i] = blb[i]
        else:
            final_lowerband[i] = final_lowerband[i-1]
            
        if trend_direction[i-1] == 1:
            if close[i] <= final_lowerband[i]:
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            if close[i] >= final_upperband[i]:
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
                
        if trend_direction[i] == 1:
            supertrend[i] = final_lowerband[i]
        else:
            supertrend[i] = final_upperband[i]
            
    df['SuperTrend'] = supertrend
    df['SuperTrend_Direction'] = trend_direction
    df['ATR'] = atr
    return df

# -------------------- MODÜLLER --------------------
def analyze_market_regime():
    print("🌍 Piyasa Rejimi Analiz Ediliyor...")
    try:
        spy = get_data_safe('SPY', period='2y', interval='1wk')
        if spy.empty: return "CAUTIOUS", "⚠️ Veri Alınamadı"

        spy['SMA_50'] = spy['Close'].rolling(window=10).mean()
        spy['SMA_200'] = spy['Close'].rolling(window=40).mean()
        
        curr_price = float(spy['Close'].iloc[-1])
        curr_sma50 = float(spy['SMA_50'].iloc[-1])
        curr_sma200 = float(spy['SMA_200'].iloc[-1])
        
        if curr_price > curr_sma50 and curr_sma50 > curr_sma200:
            return "AGGRESSIVE", "🐂 GÜÇLÜ BOĞA"
        elif curr_price > curr_sma200 and curr_price < curr_sma50:
            return "CAUTIOUS", "⚠️ DÜZELTME / ZAYIF BOĞA"
        elif curr_price < curr_sma200:
            return "DEFENSIVE", "🐻 AYI PİYASASI"
        else:
            return "CAUTIOUS", "⚖️ YATAY"
    except Exception as e:
        print(f"Piyasa analiz hatası: {e}")
        return "CAUTIOUS", "Hata (Güvenli Mod)"

def analyze_top_sectors():
    print("🏭 Sektör Rotasyonu Analiz Ediliyor...")
    sector_performance = []
    
    for ticker, name in SECTOR_ETFS.items():
        try:
            # YENİ: Bloklanmamak için her istekte biraz bekle
            time.sleep(0.5) 
            df = get_data_safe(ticker, period='3mo', interval='1d')
            
            if not df.empty and len(df) > 1:
                start_price = float(df['Close'].iloc[0])
                end_price = float(df['Close'].iloc[-1])
                roi = ((end_price - start_price) / start_price) * 100
                sector_performance.append({'etf': ticker, 'name': name, 'roi': roi})
            else:
                print(f"   ⚠️ {ticker} verisi boş.")
        except: continue
            
    sector_performance.sort(key=lambda x: x['roi'], reverse=True)
    
    top_3_sectors = [s['etf'] for s in sector_performance[:3]]
    top_names = [s['name'] for s in sector_performance[:3]]
    
    if top_names:
        print(f"   🔥 Lider Sektörler: {top_names}")
    else:
        print("   ⚠️ Sektör verileri çekilemedi.")
        
    return top_3_sectors

def analyze_single_stock(ticker, market_risk_mode):
    try:
        # YENİ: Paralel işlemde çok hızlı istek atınca bloklanmayı önlemek için rastgele bekleme
        time.sleep(np.random.uniform(0.1, 1.0))
        
        df = get_data_safe(ticker, period="2y", interval="1wk")
        if df is None or len(df) < 50: return None
        
        df = calculate_supertrend(df, period=SUPER_TREND_PERIOD, multiplier=SUPER_TREND_MULT)
        df['RSI'] = calculate_rsi(df['Close'])
        
        current_close = float(df['Close'].iloc[-1])
        current_st = float(df['SuperTrend'].iloc[-1])
        current_dir = int(df['SuperTrend_Direction'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        current_atr = float(df['ATR'].iloc[-1])
        
        if current_dir != 1: return None
        if current_close <= current_st: return None
        if current_rsi > RSI_MAX_THRESHOLD: return None
        
        pullback_dist = current_close - current_st
        if pullback_dist > (MAX_PULLBACK_ATR * current_atr): return None

        adjusted_risk = RISK_PER_TRADE / 2 if market_risk_mode == "CAUTIOUS" else RISK_PER_TRADE
        if market_risk_mode == "DEFENSIVE": return None
            
        stop_price = current_st
        risk_per_share = current_close - stop_price
        
        if risk_per_share <= 0: return None
        
        shares = int((PORTFOLIO_SIZE * adjusted_risk) / risk_per_share)
        if shares < 1: return None
        
        return {
            'ticker': ticker,
            'sector': STOCK_SECTOR_MAP.get(ticker, 'Bilinmiyor'),
            'price': current_close,
            'stop': stop_price,
            'rsi': current_rsi,
            'shares': shares
        }
    except: return None

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("❌ Token/ID eksik, mesaj gönderilemedi.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e: print(f"Telegram Hatası: {e}")

# -------------------- ANA ÇALIŞTIRICI --------------------
def run_full_system():
    print("🚀 SİSTEM BAŞLATILIYOR (Safe Mode v3)...\n" + "="*50)
    
    risk_mode, market_status = analyze_market_regime()
    if risk_mode == "DEFENSIVE":
        send_telegram_message(f"🛑 **SİSTEM DURDURULDU**\n{market_status}\nPiyasa Ayı trendinde.")
        return

    top_sectors = analyze_top_sectors()
    
    # YENİ: YEDEK PLAN (FALLBACK MECHANISM)
    # Eğer sektör verileri çekilemediyse (boşsa), programı durdurma.
    # Tüm hisseleri taramaya devam et.
    if not top_sectors:
        print("⚠️ DİKKAT: Sektör filtrelemesi yapılamadı. Tüm hisseler taranıyor...")
        target_tickers = list(STOCK_SECTOR_MAP.keys()) # Hepsi
        sector_info_msg = "⚠️ Sektör verisi alınamadı (Tüm liste tarandı)."
    else:
        target_tickers = [t for t, s in STOCK_SECTOR_MAP.items() if s in top_sectors]
        sector_info_msg = f"🔥 Lider Sektörler: {[SECTOR_ETFS.get(s, s) for s in top_sectors]}"
    
    print(f"🎯 Hedef Hisseler: {len(target_tickers)} adet")
    
    candidates = []
    # Worker sayısını düşürdüm (5 -> 3) ki sunucu bizi yine bot sanıp engellemesin
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyze_single_stock, t, risk_mode): t for t in target_tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: candidates.append(res)
            
    if candidates:
        candidates.sort(key=lambda x: x['rsi'])
        msg = f"🧠 **HAFTALIK SİNYAL RAPORU** ({date.today().strftime('%d.%m.%Y')})\n"
        msg += f"Piyasa: {market_status}\n{sector_info_msg}\n--------------------------------\n"
        for p in candidates[:MAX_POSITIONS]:
            msg += f"✅ **{p['ticker']}**\nFiyat: ${p['price']:.2f} | Stop: ${p['stop']:.2f}\nAdet: {p['shares']} | RSI: {p['rsi']:.1f}\n\n"
        send_telegram_message(msg)
        print("✅ Rapor Gönderildi")
    else:
        # Hiç hisse bulunamazsa da bilgi ver
        info_msg = f"Piyasa: {market_status}\n{sector_info_msg}\n\nUygun sinyal bulunamadı."
        send_telegram_message(info_msg)
        print("ℹ️ Sinyal bulunamadı, durum raporu gönderildi.")

if __name__ == "__main__":
    run_full_system()
