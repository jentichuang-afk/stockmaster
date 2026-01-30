import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai

# --- 🔴 診斷代碼 (請貼在 import 之後，頁面設定之前) ---
import google.generativeai as genai
import streamlit as st

st.error(f"🔍 目前安裝的 AI 套件版本: {genai.__version__}")

try:
    st.write("🔑 您的 API Key 目前能使用的模型清單：")
    for m in genai.list_models():
        if 'gemini' in m.name:
            st.code(m.name)
except Exception as e:
    st.error(f"❌ 連線測試失敗: {e}")
# ----------------------------------------------------

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：真·AI 戰情室", layout="wide", page_icon="🧠")
st.title("🧠 股票大師：真·AI 戰情室 (Powered by Gemini)")

# --- 安全性設定：嘗試讀取 API Key ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') # 使用快速且免費額度較高的模型
    ai_available = True
except Exception as e:
    ai_available = False
    st.warning("⚠️ 未偵測到 Gemini API Key，將切換回傳統規則式分析。請在 Streamlit Secrets 設定 GEMINI_API_KEY。")

# --- 2. 側邊欄參數 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入股票代碼", value="2330", help="台股請輸入如 2330, 美股如 NVDA")
days_input = st.sidebar.slider("K線觀察天數", 60, 730, 180)

st.sidebar.subheader("📊 技術指標開關")
show_ma = st.sidebar.checkbox("顯示均線 (MA)", value=True)
show_bb = st.sidebar.checkbox("顯示布林通道", value=True)
show_kd = st.sidebar.checkbox("顯示 KD", value=True)
show_macd = st.sidebar.checkbox("顯示 MACD", value=True)
show_obv = st.sidebar.checkbox("顯示 OBV", value=True)

run_btn = st.sidebar.button("🚀 呼叫 Gemini 進行分析", type="primary")

# --- 3. 核心函數：計算指標 ---
def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()

    # 布林
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    k_list = [50]; d_list = [50]
    for r in df['RSV']:
        if pd.isna(r): k_list.append(50); d_list.append(50)
        else:
            k = (2/3) * k_list[-1] + (1/3) * r
            d = (2/3) * d_list[-1] + (1/3) * k
            k_list.append(k); d_list.append(d)   
    df['K'] = k_list[1:]; df['D'] = d_list[1:]

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    return df

# --- 4. 核心函數：呼叫 Gemini AI 進行深度分析 ---
def call_gemini_analysis(df, info, symbol):
    # 準備數據摘要 (只取最近 5 天的數據給 AI，避免 token 太多)
    recent_data = df.tail(5).to_string()
    
    last = df.iloc[-1]
    
    # 準備基本面數據字串
    pe = info.get('trailingPE', 'N/A')
    roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 'N/A'
    peg = info.get('pegRatio', 'N/A')
    
    # 建立 Prompt (提示詞)
    prompt = f"""
    你是一位華爾街頂級操盤手與分析師。請根據以下數據，對股票代號 {symbol} 進行深度技術面與基本面分析。
    
    【基本面數據】
    本益比(PE): {pe}, ROE: {roe}%, PEG: {peg}
    
    【近五日技術指標數據 (包含 K值, D值, MACD, OBV, 布林通道)】
    {recent_data}
    
    請依據上述數據，使用繁體中文回答，包含以下三個部分：
    1. **趨勢與籌碼判讀**：觀察均線排列、MACD 柱狀體變化、OBV 能量潮是否有背離或是主力進貨。
    2. **操作策略建議**：現在適合「買進」、「賣出」還是「觀望」？請給出具體的支撐位與壓力位建議。
    3. **綜合評分**：請根據基本面與技術面，給出一個 0-100 的分數，並簡述理由。
    
    請注意：回答要專業、犀利，直接切入重點，不要講模稜兩可的廢話。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 連線分析失敗：{str(e)}。請檢查 API Key 是否正確或額度是否足夠。"

# --- 5. 主程式 ---
if run_btn and ticker_input:
    symbol = ticker_input.strip().upper()
    if symbol.isdigit(): symbol += ".TW"
    
    with st.spinner(f"正在連線 Gemini 大腦分析 {symbol} ..."):
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
        
        # --- 看板區 ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("最新股價", f"{last_close:.2f}", f"{chg:.2f} ({pct:.2f}%)")
        
        pe = info.get('trailingPE', 'N/A')
        pb = info.get('priceToBook', 'N/A')
        peg = info.get('pegRatio', 'N/A')
        
        col2.metric("PE", f"{pe}")
        col3.metric("PB", f"{pb}")
        col4.metric("PEG", f"{peg}")
        col5.metric("成交量", f"{int(df['Volume'].iloc[-1]/1000)}張")

        # --- 分頁 ---
        tab1, tab2, tab3 = st.tabs(["📊 互動圖表", "🏢 財報數據", "🧠 Gemini 深度解盤"])

        with tab1:
            rows = 2
            row_heights = [0.5, 0.15]
            indicators_to_plot = []
            if show_macd: indicators_to_plot.append('MACD')
            if show_obv: indicators_to_plot.append('OBV')
            if show_kd: indicators_to_plot.append('KD')
            for _ in indicators_to_plot:
                rows += 1; row_heights.append(0.15)
                
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)
            
            # K線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            if show_ma:
                fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='blue', width=1), name='MA5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
            if show_bb:
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='布林上'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='布林下', fill='tonexty'), row=1, col=1)

            # 成交量
            colors = ['red' if o < c else 'green' for o, c in zip(df['Open'], df['Close'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
            
            current_row = 3
            if show_macd:
                hist_colors = ['red' if h > 0 else 'green' for h in df['MACD_Hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_colors, name='MACD柱'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange', width=1), name='DIF'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='blue', width=1), name='MACD'), row=current_row, col=1)
                current_row += 1
            if show_obv:
                fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='purple', width=2), name='OBV', fill='tozeroy'), row=current_row, col=1)
                current_row += 1
            if show_kd:
                fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='orange', width=1), name='K'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='purple', width=1), name='D'), row=current_row, col=1)

            fig.update_layout(height=900, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if not financials.empty:
                st.dataframe(financials)
            else:
                st.warning("無財報資料")

        with tab3:
            st.subheader(f"🧠 Gemini 對 {symbol} 的深度分析")
            if ai_available:
                # 這裡呼叫真 AI
                ai_response = call_gemini_analysis(df, info, symbol)
                st.markdown(ai_response)
            else:
                st.error("請先設定 GEMINI_API_KEY 才能使用此功能。")

else:
    st.info("👈 請在側邊欄輸入股票代碼並按下按鈕")







