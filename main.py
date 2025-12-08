import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# ta kütüphanesi çıkarıldı, formülleri aşağıda manuel yazdık.

warnings.filterwarnings('ignore')

# -------------------- AYARLAR --------------------
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN" 
CHAT_ID = "YOUR_CHAT_ID"

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
SECTOR_ETFS = {
    'XLK': 'Teknoloji', 'XLF': 'Finans', 'XLV': 'Sağlık', 'XLE': 'Enerji',
    'XLY': 'Tüketim (Lüks)', 'XLP': 'Tüketim (Temel)', 'XLI': 'Sanayi',
    'XLC': 'İletişim', 'XLB': 'Hammadde', 'XLRE': 'Gayrimenkul', 'XLU': 'Altyapı'
}

STOCK_SECTOR_MAP = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'ADBE': 'XLK', 'ORCL': 'XLK',
    'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'T': 'XLC', 'VZ': 'XLC',
    'AMZN': 'XLY', 'TSLA': 'XLY', 'MCD': 'XLY', 'NKE': 'XLY',
    'JPM': 'XLF', 'V': 'XLF', 'MA': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'GS': 'XLF',
    'JNJ': 'XLV', 'UNH': 'XLV', 'PFE': 'XLV', 'LLY': 'XLV', 'ABBV': 'XLV',
    'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE',
    'CAT': 'XLI', 'BA': 'XLI', 'HON': 'XLI', 'GE': 'XLI',
    'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP', 'WMT': 'XLP', 'COST': 'XLP'
}

# -------------------- MANUEL İNDİKATÖR HESAPLAMALARI --------------------

def calculate_atr(df, period=14):
    """Average True Range (ATR) hesaplar"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean()

def calculate_rsi(series, period=14):
    """Relative Strength Index (RSI) hesaplar"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_supertrend(df, period=10, multiplier=3):
    """SuperTrend İndikatörünü manuel hesaplar"""
    atr = calculate_atr(df, period)
    
    hl2 = (df['High'] + df['Low']) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    # Hesaplama döngüsü (Pandas ile vectorize etmek zordur, döngü daha güvenli)
    final_upperband = [0.0] * len(df)
    final_lowerband = [0.0] * len(df)
    supertrend = [0.0] * len(df)
    trend_direction = [1] * len(df) # 1: Up, -1: Down
    
    close = df['Close'].values
    
    for i in range(1, len(df)):
        # Upper Band Logic
        if basic_upperband.iloc[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
            final_upperband[i] = basic_upperband.iloc[i]
        else:
            final_upperband[i] = final_upperband[i-1]
            
        # Lower Band Logic
        if basic_lowerband.iloc[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
            final_lowerband[i] = basic_lowerband.iloc[i]
        else:
            final_lowerband[i] = final_lowerband[i-1]
            
        # Trend Direction
        if trend_direction[i-1] == 1: # Trend yukarıysa
            if close[i] <= final_lowerband[i]:
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else: # Trend aşağıysa
            if close[i] >= final_upperband[i]:
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
                
        # SuperTrend Value
        if trend_direction[i] == 1:
            supertrend[i] = final_lowerband[i]
        else:
            supertrend[i] = final_upperband[i]
            
    df['SuperTrend'] = supertrend
    df['SuperTrend_Direction'] = trend_direction
    df['ATR'] = atr
    
    return df

# -------------------- 1. MODÜL: PİYASA ANALİZİ --------------------
def analyze_market_regime():
    print("🌍 Piyasa Rejimi Analiz Ediliyor...")
    try:
        spy = yf.download('SPY', period='2y', interval='1wk', progress=False)
        
        # Basit Hareketli Ortalamalar (Pandas ile)
        spy['SMA_50'] = spy['Close'].rolling(window=10).mean() # Haftalıkta 10 bar ~ 50 Gün
        spy['SMA_200'] = spy['Close'].rolling(window=40).mean() 
        
        curr_price = spy['Close'].iloc[-1]
        curr_sma50 = spy['SMA_50'].iloc[-1]
        curr_sma200 = spy['SMA_200'].iloc[-1]
        
        if curr_price > curr_sma50 and curr_sma50 > curr_sma200:
            return "AGGRESSIVE", "🐂 GÜÇLÜ BOĞA"
        elif curr_price > curr_sma200 and curr_price < curr_sma50:
            return "CAUTIOUS", "⚠️ DÜZELTME / ZAYIF BOĞA"
        elif curr_price < curr_sma200:
            return "DEFENSIVE", "🐻 AYI PİYASASI"
        else:
            return "CAUTIOUS", "⚖️ YATAY"
            
    except Exception as e:
        logging.error(f"Piyasa analizi hatası: {e}")
        return "CAUTIOUS", "Hata Oluştu"

# -------------------- 2. MODÜL: SEKTÖR ANALİZİ --------------------
def analyze_top_sectors():
    print("🏭 Sektör Rotasyonu Analiz Ediliyor...")
    sector_performance = []
    
    for ticker, name in SECTOR_ETFS.items():
        try:
            df = yf.download(ticker, period='3mo', interval='1d', progress=False)
            if len(df) > 0:
                roi = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                sector_performance.append({'etf': ticker, 'name': name, 'roi': roi})
        except: continue
            
    sector_performance.sort(key=lambda x: x['roi'], reverse=True)
    top_3_sectors = [s['etf'] for s in sector_performance[:3]]
    print(f"   🔥 Lider Sektörler: {[s['name'] for s in sector_performance[:3]]}")
    return top_3_sectors

# -------------------- 3. MODÜL: HİSSE ANALİZİ --------------------
def analyze_single_stock(ticker, market_risk_mode):
    try:
        df = yf.download(ticker, period="2y", interval="1wk", progress=False)
        if df is None or len(df) < 50: return None
        
        # Manuel yazdığımız fonksiyonları çağırıyoruz
        df = calculate_supertrend(df, period=SUPER_TREND_PERIOD, multiplier=SUPER_TREND_MULT)
        df['RSI'] = calculate_rsi(df['Close'])
        
        current = df.iloc[-1]
        
        # --- FİLTRELER ---
        if current['SuperTrend_Direction'] != 1: return None
        if current['Close'] <= current['SuperTrend']: return None
        if current['RSI'] > RSI_MAX_THRESHOLD: return None
        
        # Pullback Kontrolü
        pullback_dist = current['Close'] - current['SuperTrend']
        if pullback_dist > (MAX_PULLBACK_ATR * current['ATR']): return None

        # --- RİSK YÖNETİMİ ---
        adjusted_risk = RISK_PER_TRADE / 2 if market_risk_mode == "CAUTIOUS" else RISK_PER_TRADE
        if market_risk_mode == "DEFENSIVE": return None
            
        stop_price = current['SuperTrend']
        risk_per_share = current['Close'] - stop_price
        
        if risk_per_share <= 0: return None
        
        shares = int((PORTFOLIO_SIZE * adjusted_risk) / risk_per_share)
        if shares < 1: return None
        
        return {
            'ticker': ticker,
            'sector': STOCK_SECTOR_MAP.get(ticker, 'Bilinmiyor'),
            'price': current['Close'],
            'stop': stop_price,
            'rsi': current['RSI'],
            'shares': shares,
            'position_value': shares * current['Close']
        }
        
    except Exception as e:
        return None

# -------------------- TELEGRAM --------------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except: pass

# -------------------- ANA ÇALIŞTIRICI --------------------
def run_full_system():
    print("🚀 SİSTEM BAŞLATILIYOR (No-TA Library Version)...\n" + "="*50)
    
    risk_mode, market_status = analyze_market_regime()
    if risk_mode == "DEFENSIVE":
        send_telegram_message(f"🛑 **SİSTEM DURDURULDU**\n{market_status}")
        return

    top_sectors = analyze_top_sectors()
    target_tickers = [t for t, s in STOCK_SECTOR_MAP.items() if s in top_sectors]
    
    candidates = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_single_stock, t, risk_mode): t for t in target_tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: candidates.append(res)
            
    if candidates:
        candidates.sort(key=lambda x: x['rsi'])
        msg = f"🧠 **HAFTALIK RAPOR** ({date.today().strftime('%d.%m.%Y')})\n"
        msg += f"Piyasa: {market_status}\n--------------------------------\n"
        for p in candidates[:MAX_POSITIONS]:
            msg += f"✅ **{p['ticker']}**\nFiyat: ${p['price']:.2f} | Stop: ${p['stop']:.2f}\nAdet: {p['shares']} | RSI: {p['rsi']:.1f}\n\n"
        send_telegram_message(msg)
        print("✅ Rapor Gönderildi")
    else:
        send_telegram_message(f"Piyasa: {market_status}\nUygun hisse bulunamadı.")
        print("ℹ️ Hisse bulunamadı")

if __name__ == "__main__":
    run_full_system()
