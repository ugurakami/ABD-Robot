import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
import os
import time
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
MAX_PULLBACK_ATR = 3.0   # Boğa piyasasında toleransı artırdık
RSI_MAX_THRESHOLD = 80   # Boğa piyasasında RSI 80'e kadar izin verdik

# -------------------- AKILLI VERİTABANI (MASTER DATABASE) --------------------
# Artık elle tek tek hisse girmene gerek yok.
# Sektör lideri seçildiğinde, sistem buradaki "En Babalar" listesini tarayacak.

SECTOR_HOLDINGS = {
    # 1. TEKNOLOJİ (XLK) - En büyükler
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'TXN', 'INTC', 'IBM', 'NOW', 'AMAT', 'MU'],
    
    # 2. İLETİŞİM (XLC) - Google, Meta vs.
    'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'TMUS', 'CMCSA', 'DIS', 'T', 'VZ', 'CHTR'],
    
    # 3. FİNANS (XLF) - Bankalar ve Kredi Kartları
    'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'AXP', 'BLK', 'C', 'SPGI'],
    
    # 4. SAĞLIK (XLV) - İlaç devleri
    'XLV': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'AMGN', 'PFE', 'ISRG', 'DHR', 'BMY'],
    
    # 5. TÜKETİM LÜKS (XLY) - Amazon, Tesla
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TJX', 'MAR'],
    
    # 6. ENERJİ (XLE) - Petrol devleri
    'XLE': ['XOM', 'CVX', 'EOG', 'COP', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    
    # 7. SANAYİ (XLI)
    'XLI': ['CAT', 'GE', 'UNP', 'HON', 'RTX', 'LMT', 'DE', 'UPS', 'BA', 'ADP'],
    
    # 8. TÜKETİM TEMEL (XLP) - Market ürünleri
    'XLP': ['PG', 'COST', 'WMT', 'PEP', 'KO', 'PM', 'MO', 'CL', 'TGT'],
    
    # 9. ALTIN VE MADENCİLİK (GLD) - ***KRİTİK GÜNCELLEME***
    # Altın lider çıkarsa sadece ETF değil, madencileri de tarayacak.
    'GLD': ['GLD', 'NEM', 'GOLD', 'AEM', 'RGLD', 'FNV', 'KGC', 'AU'],
    
    # 10. TAHVİL VE KORUMA (TLT)
    'TLT': ['TLT', 'TMF', 'SHY', 'IEF'],
    
    # 11. GENEL ENDEKS
    'SPY': ['SPY', 'QQQ', 'DIA', 'IWM']
}

SECTOR_NAMES = {
    'XLK': 'Teknoloji', 'XLF': 'Finans', 'XLV': 'Sağlık', 'XLE': 'Enerji',
    'XLY': 'Lüks Tüketim', 'XLP': 'Temel Tüketim', 'XLI': 'Sanayi',
    'XLC': 'İletişim', 'GLD': 'ALTIN', 'TLT': 'Tahvil', 'SPY': 'Genel Piyasa'
}

# -------------------- YARDIMCI FONKSİYONLAR --------------------

def get_data_safe(ticker, period, interval):
    """Bot korumasını aşmak için güvenli veri çekme"""
    try:
        dat = yf.Ticker(ticker)
        df = dat.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            df = yf.download(ticker, period=period, interval=interval, progress=False, ignore_tz=True)
        return df
    except: return pd.DataFrame()

def calculate_supertrend(df, period=10, multiplier=3):
    # ATR Hesapla
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    atr = np.max(ranges, axis=1).rolling(period).mean()
    
    # SuperTrend Hesapla
    hl2 = (df['High'] + df['Low']) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    # Numpy dizilerine çevir (Hız için)
    close = df['Close'].values
    bub, blb = basic_upperband.values, basic_lowerband.values
    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.ones(len(df))
    
    for i in range(1, len(df)):
        if bub[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]: final_upper[i] = bub[i]
        else: final_upper[i] = final_upper[i-1]
            
        if blb[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]: final_lower[i] = blb[i]
        else: final_lower[i] = final_lower[i-1]
            
        if direction[i-1] == 1:
            if close[i] <= final_lower[i]: direction[i] = -1
            else: direction[i] = 1
        else:
            if close[i] >= final_upper[i]: direction[i] = 1
            else: direction[i] = -1
            
        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
            
    df['SuperTrend'] = supertrend
    df['SuperTrend_Direction'] = direction
    df['ATR'] = atr
    
    # RSI Hesapla
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# -------------------- ANA ANALİZ MODÜLLERİ --------------------

def analyze_market_regime():
    print("🌍 Piyasa Rejimi Analiz Ediliyor...")
    try:
        spy = get_data_safe('SPY', period='2y', interval='1wk')
        if spy.empty: return "CAUTIOUS", "Veri Yok"
        
        spy['SMA50'] = spy['Close'].rolling(10).mean()
        spy['SMA200'] = spy['Close'].rolling(40).mean()
        
        curr = float(spy['Close'].iloc[-1])
        sma50 = float(spy['SMA50'].iloc[-1])
        sma200 = float(spy['SMA200'].iloc[-1])
        
        if curr > sma50 > sma200: return "AGGRESSIVE", "🐂 GÜÇLÜ BOĞA"
        elif curr > sma200: return "CAUTIOUS", "⚠️ DÜZELTME / ZAYIF BOĞA"
        else: return "DEFENSIVE", "🐻 AYI PİYASASI"
    except: return "CAUTIOUS", "Hata"

def analyze_top_sectors():
    print("🏭 Sektör Liderleri Taranıyor...")
    results = []
    
    # Sadece anahtar (ETF) isimlerini dön
    for etf in SECTOR_HOLDINGS.keys():
        if etf == 'SPY': continue # SPY'ı sıralamaya sokma, o genel piyasa
        try:
            time.sleep(0.2)
            df = get_data_safe(etf, period='3mo', interval='1d')
            if len(df) > 0:
                roi = ((float(df['Close'].iloc[-1]) - float(df['Close'].iloc[0])) / float(df['Close'].iloc[0])) * 100
                results.append({'etf': etf, 'roi': roi})
        except: continue
            
    results.sort(key=lambda x: x['roi'], reverse=True)
    top_3 = [x['etf'] for x in results[:3]]
    
    print(f"   🔥 Liderler: {[SECTOR_NAMES.get(x, x) for x in top_3]}")
    return top_3

def analyze_stock(ticker, sector_name, risk_mode):
    try:
        time.sleep(np.random.uniform(0.1, 0.5))
        df = get_data_safe(ticker, period='2y', interval='1wk')
        if df is None or len(df) < 50: return None
        
        df = calculate_supertrend(df)
        curr = df.iloc[-1]
        
        # --- FİLTRELER ---
        if curr['SuperTrend_Direction'] != 1: return None
        if curr['Close'] <= curr['SuperTrend']: return None
        if curr['RSI'] > RSI_MAX_THRESHOLD: return None # RSI 80 Filtresi
        
        atr_val = float(curr['ATR'])
        if (curr['Close'] - curr['SuperTrend']) > (MAX_PULLBACK_ATR * atr_val): return None
        
        # --- RISK HESABI ---
        risk_factor = RISK_PER_TRADE / 2 if risk_mode == "CAUTIOUS" else RISK_PER_TRADE
        if risk_mode == "DEFENSIVE": return None
        
        stop = float(curr['SuperTrend'])
        risk_share = curr['Close'] - stop
        if risk_share <= 0: return None
        
        shares = int((PORTFOLIO_SIZE * risk_factor) / risk_share)
        if shares < 1: return None
        
        return {
            'ticker': ticker,
            'sector': sector_name,
            'price': float(curr['Close']),
            'stop': stop,
            'rsi': float(curr['RSI']),
            'shares': shares
        }
    except: return None

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID: 
        print(msg)
        return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                       json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# -------------------- SİSTEMİ ÇALIŞTIR --------------------
def run():
    print("🚀 AKILLI ROBOT BAŞLATILIYOR...\n" + "="*50)
    
    risk_mode, market_status = analyze_market_regime()
    if risk_mode == "DEFENSIVE":
        send_telegram(f"🛑 **SİSTEM BEKLEMEDE**\n{market_status}\nNakit candır.")
        return

    top_sectors = analyze_top_sectors()
    
    # --- AKILLI HİSSE SEÇİMİ ---
    # Lider sektörlerin içindeki hisseleri havuza ekle
    target_pool = []
    
    for sector in top_sectors:
        holdings = SECTOR_HOLDINGS.get(sector, [])
        sector_nice_name = SECTOR_NAMES.get(sector, sector)
        print(f"   🔍 {sector_nice_name} içindeki {len(holdings)} hisse taranıyor...")
        
        for ticker in holdings:
            target_pool.append((ticker, sector_nice_name))
            
    # Tekrarları önle (Set yapısı)
    target_pool = list(set(target_pool))
    
    print(f"🎯 Toplam Taranacak Aday: {len(target_pool)}")
    
    candidates = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(analyze_stock, t, s, risk_mode): t for t, s in target_pool}
        for f in as_completed(futures):
            res = f.result()
            if res: candidates.append(res)
            
    if candidates:
        candidates.sort(key=lambda x: x['rsi']) # Düşük RSI (Daha çok potansiyel) en üste
        msg = f"🧠 **ALIM SİNYALLERİ** ({date.today().strftime('%d.%m.%Y')})\n"
        msg += f"Durum: {market_status}\n"
        msg += f"Liderler: {[SECTOR_NAMES.get(x, x) for x in top_sectors]}\n"
        msg += "-"*30 + "\n"
        
        for c in candidates[:MAX_POSITIONS]:
            msg += f"✅ **{c['ticker']}** ({c['sector']})\n"
            msg += f"Fiyat: ${c['price']:.2f} | Stop: ${c['stop']:.2f}\n"
            msg += f"Adet: {c['shares']} | RSI: {c['rsi']:.1f}\n\n"
            
        send_telegram(msg)
        print("✅ Sinyaller gönderildi.")
    else:
        send_telegram(f"Durum: {market_status}\nLider sektörlerde teknik kriterlere uyan (ucuz kalmış) hisse bulunamadı.")
        print("ℹ️ Sinyal yok.")

if __name__ == "__main__":
    run()
