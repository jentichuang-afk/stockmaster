import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：法人操盤室", layout="wide", page_icon="🏦")
st.title("🏦 股票大師：法人操盤室 (Institutional Edition)")

# --- 2. 側邊欄參數 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入股票代碼", value="2330", help="台股請輸入如 2330, 美股如 NVDA")
days_input = st.sidebar.slider("K線觀察天數", 60, 730, 180)

st.sidebar.subheader("📊 技術指標開關")
show_ma = st.sidebar.checkbox("顯示均線 (MA)", value=True)
show_bb = st.sidebar.checkbox("顯示布林通道", value=True)
show_kd = st.sidebar.checkbox("顯示 KD (短線)", value=True)
show_macd = st.sidebar.checkbox("顯示 MACD (波段)", value=True)
show_obv = st.sidebar.checkbox("顯示 OBV (籌碼)", value=True)

run_btn = st.sidebar.button("🚀 啟動法人級分析", type="primary")

# --- 3. 核心函數：計算全方位指標 ---
def calculate_indicators(df):
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()

    # 布林通道
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    # KD 指標
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    
    k_list = [50]
    d_list = [50]
    for r in df['RSV']:
        if pd.isna(r):
            k_list.append(k_list[-1])
            d_list.append(d_list[-1])
        else:
            k = (2/3) * k_list[-1] + (1/3) * r
            d = (2/3) * d_list[-1] + (1/3) * k
            k_list.append(k)
            d_list.append(d)   
    df['K'] = k_list[1:]
    df['D'] = d_list[1:]

    # MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # OBV (On Balance Volume) - 能量潮
    # OBV = 前日OBV + (若今日漲: +成交量, 若今日跌: -成交量)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    return df

# --- 4. 核心函數：AI 綜合分析 (加入 MACD 與 OBV) ---
def generate_ai_report(df, info, symbol):
    report = []
    score = 50 
    
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # === A. 技術面深度診斷 ===
    report.append("### 📈 1. 技術籌碼雙重診斷")
    
    # 1. 趨勢 (MA)
    if last['MA5'] > last['MA20'] > last['MA60']:
        report.append("✅ **【趨勢：多頭排列】**：短中長均線向上，主力控盤穩固。 (+15分)")
        score += 15
    elif last['MA5'] < last['MA20'] < last['MA60']:
        report.append("❄️ **【趨勢：空頭排列】**：均線反壓，上方套牢重重。 (-15分)")
        score -= 15
    
    # 2. 波段 (MACD) - 新增
    if last['MACD_Hist'] > 0 and prev['MACD_Hist'] < 0:
        report.append("🐂 **【MACD：柱狀體翻紅】**：空轉多關鍵時刻，波段買點浮現！ (+15分)")
        score += 15
    elif last['MACD'] > last['Signal'] and last['MACD'] > 0:
        report.append("🚀 **【MACD：黃金交叉且在零軸上】**：多頭趨勢正在加速中。 (+10分)")
        score += 10
    elif last['MACD'] < last['Signal']:
        report.append("📉 **【MACD：死亡交叉】**：波段動能轉弱，留意修正。 (-5分)")
        score -= 5

    # 3. 籌碼 (OBV) - 新增 (非常專業的判斷)
    # 簡單判斷：最近5天 OBV 趨勢
    obv_trend = df['OBV'].iloc[-5:].mean() > df['OBV'].iloc[-10:-5].mean()
    price_trend = df['Close'].iloc[-5:].mean() > df['Close'].iloc[-10:-5].mean()

    if obv_trend and not price_trend:
        report.append("🕵️ **【OBV：主力吸籌】**：股價未漲但 OBV 先行向上，大戶正在偷偷進貨！ (+20分)")
        score += 20
    elif not obv_trend and price_trend:
        report.append("⚠️ **【OBV：量價背離】**：股價創高但 OBV 沒跟上，小心主力拉高出貨。 (-15分)")
        score -= 15
    elif obv_trend and price_trend:
        report.append("💰 **【OBV：量價齊揚】**：買氣充足，籌碼安定。 (+5分)")
        score += 5

    # 4. KD (短線)
    if last['K'] < 20 and last['K'] > last['D'] and prev['K'] < prev['D']:
        report.append("💎 **【KD：低檔黃金交叉】**：短線超賣後的反彈訊號。 (+10分)")
        score += 10

    # === B. 基本面體質分析 ===
    report.append("### 🏢 2. 價值與體質檢測")
    
    # ROE
    roe = info.get('returnOnEquity', 0)
    if roe and roe > 0.15:
        report.append(f"👑 **【高 ROE】**：ROE {roe*100:.1f}%，頂級賺錢體質。 (+10分)")
        score += 10
        
    # 本益比
    pe = info.get('trailingPE')
    if pe and pe < 15:
        report.append(f"💰 **【低本益比】**：PE {pe:.1f} 倍，股價相對便宜。 (+10分)")
        score += 10
    
    # PEG (成長估值) - 新增
    peg = info.get('pegRatio')
    if peg:
        if peg < 1:
            report.append(f"🦄 **【PEG < 1】**：成長力道強於估值，這是彼得林區最愛的飆股特徵！ (+15分)")
            score += 15
        elif peg > 2:
            report.append(f"🎈 **【PEG > 2】**：股價成長溢價過高，買進風險增加。 (-5分)")
            score -= 5

    score = max(0, min(100, score))
    return report, score

# --- 5. 主程式 ---
if run_btn and ticker_input:
    symbol = ticker_input.strip().upper()
    if symbol.isdigit(): symbol += ".TW"
    
    with st.spinner(f"正在調用法人級數據： {symbol} ..."):
        try:
            end = datetime.now()
            start = end - timedelta(days=days_input + 100)
            df_raw = yf.download(symbol, start=start, end=end, progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            
            stock = yf.Ticker(symbol)
            info = stock.info
            financials = stock.financials
            
        except Exception as e:
            st.error(f"數據抓取失敗：{e}")
            df_raw = pd.DataFrame()

    if df_raw.empty:
        st.error("❌ 找不到資料，請檢查代碼。")
    else:
        # 計算指標
        df = calculate_indicators(df_raw).iloc[-days_input:]
        last_close = df['Close'].iloc[-1]
        chg = last_close - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        
        # --- 看板區 (新增 PB, PEG) ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("最新股價", f"{last_close:.2f}", f"{chg:.2f} ({pct:.2f}%)")
        
        # 數據格式修正
        pe = info.get('trailingPE', 'N/A')
        pb = info.get('priceToBook', 'N/A') # 股價淨值比
        peg = info.get('pegRatio', 'N/A') # 成長估值比
        
        raw_yield = info.get('dividendYield', 0)
        fmt_yield = f"{raw_yield:.2f}%" if raw_yield > 1 else f"{raw_yield*100:.2f}%" if raw_yield else "N/A"

        col2.metric("本益比 (PE)", f"{pe}")
        col3.metric("股價淨值比 (PB)", f"{pb}")
        col4.metric("PEG (成長)", f"{peg}")
        col5.metric("殖利率", fmt_yield)

        # --- 分頁系統 ---
        tab1, tab2, tab3 = st.tabs(["📊 法人級圖表", "🏢 財報數據", "🤖 AI 深度解盤"])

        # === Tab 1: 法人級圖表 ===
        with tab1:
            # 動態計算需要的行數
            rows = 2
            row_heights = [0.5, 0.15] # K線, 量
            
            indicators_to_plot = []
            if show_macd: indicators_to_plot.append('MACD')
            if show_obv: indicators_to_plot.append('OBV')
            if show_kd: indicators_to_plot.append('KD')
            
            for _ in indicators_to_plot:
                rows += 1
                row_heights.append(0.15)
                
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)
            
            # Row 1: K線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            if show_ma:
                fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='blue', width=1), name='MA5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
            if show_bb:
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='布林上'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='布林下', fill='tonexty'), row=1, col=1)

            # Row 2: 成交量
            colors = ['red' if o < c else 'green' for o, c in zip(df['Open'], df['Close'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
            
            # 動態繪製指標
            current_row = 3
            
            if show_macd:
                # MACD 柱狀體顏色
                hist_colors = ['red' if h > 0 else 'green' for h in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_colors, name='MACD柱狀'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange', width=1), name='DIF'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='blue', width=1), name='MACD'), row=current_row, col=1)
                current_row += 1
                
            if show_obv:
                fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='purple', width=2), name='OBV能量潮', fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'), row=current_row, col=1)
                current_row += 1
                
            if show_kd:
                fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='orange', width=1), name='K值'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='purple', width=1), name='D值'), row=current_row, col=1)
                fig.add_hline(y=80, line_dash="dot", row=current_row, col=1, line_color="red")
                fig.add_hline(y=20, line_dash="dot", row=current_row, col=1, line_color="green")

            fig.update_layout(height=900, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # === Tab 2: 基本面 ===
        with tab2:
            if not financials.empty:
                fin_data = financials.T.iloc[::-1]
                rev_col = [c for c in fin_data.columns if 'Total Revenue' in c or 'Revenue' in c]
                net_col = [c for c in fin_data.columns if 'Net Income' in c]
                
                if rev_col and net_col:
                    fig_fin = go.Figure()
                    fig_fin.add_trace(go.Bar(x=fin_data.index.astype(str), y=fin_data[rev_col[0]], name='總營收', marker_color='#3366CC'))
                    fig_fin.add_trace(go.Bar(x=fin_data.index.astype(str), y=fin_data[net_col[0]], name='淨利', marker_color='#109618'))
                    fig_fin.update_layout(title="營收獲利趨勢", height=400)
                    st.plotly_chart(fig_fin, use_container_width=True)
                st.dataframe(financials)
            else:
                st.warning("無財報資料")

        # === Tab 3: AI 解盤 ===
        with tab3:
            report_lines, score = generate_ai_report(df, info, symbol)
            score_color = "green" if score > 70 else "red" if score < 40 else "orange"
            st.markdown(f"""
            <div style="text-align: center;">
                <h2>🛡️ 法人綜合評分</h2>
                <h1 style="color: {score_color}; font-size: 60px;">{score} 分</h1>
            </div>
            <hr>
            """, unsafe_allow_html=True)
            for line in report_lines:
                st.markdown(line)

else:
    st.info("👈 請在側邊欄輸入股票代碼並按下按鈕")
