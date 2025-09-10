import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# IMPORTANT: set_page_config must be the FIRST Streamlit command in the script
st.set_page_config(
    page_title="TrendPredict AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# External libraries
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# Alpha Vantage API key (set in Streamlit secrets for deployment)
try:
    ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except Exception:
    ALPHA_VANTAGE_API_KEY = "L03O604P4FDTZUWQ"  # Demo key with limited rate
AV_BASE_URL = 'https://www.alphavantage.co/query'

# Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --primary-color: #007BFF;
        --primary-color-glow: rgba(0, 123, 255, 0.5);
        --gradient-start: #007BFF;
        --gradient-end: #00C6FF;
        --bg-color: #FFFFFF;
        --secondary-bg-color: #F0F2F6;
        --widget-bg-color: #E8EBF2;
        --text-color: #1A1A1A;
        --secondary-text-color: #5C5C5C;
        --border-color: #D3D3D3;
        --border-color-light: #C0C0C0;
        --positive-color: #16A34A;
        --negative-color: #DC2626;
    }

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    h1, h2, h3 { color: var(--text-color); }
    h2 {
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg-color);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] h3 {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    .card {
        background-color: var(--secondary-bg-color);
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease-in-out;
    }
    .card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 15px var(--primary-color-glow);
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: var(--widget-bg-color);
        height: 100%;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: var(--secondary-text-color);
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .metric-card p {
        font-size: 1.1rem;
        color: var(--text-color);
        font-weight: 400;
    }
    .positive { color: var(--positive-color); }
    .negative { color: var(--negative-color); }
    .stButton > button {
        background: linear-gradient(90deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 10px var(--primary-color-glow);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--widget-bg-color);
        border-radius: 8px;
        border: 1px solid var(--border-color-light);
        font-weight: 600;
        color: var(--secondary-text-color);
        padding: 0.7rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
        border: 1px solid var(--primary-color);
    }
    .warning-card {
        background-color: #FFFBEA;
        border-left: 5px solid #F9D14C;
        padding: 1rem;
        border-radius: 4px;
        color: #795514;
    }
            
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}       
</style>
""", unsafe_allow_html=True)

MARKET_STOCKS = {
    "🇺🇸 US Markets": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet (Google)",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "META": "Meta Platforms",
        "NVDA": "NVIDIA"
    },
    "🇺🇸 US Finance": {
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "BAC": "Bank of America Corp",
        "WFC": "Wells Fargo & Company"
    },
    "🇮🇳 Indian Markets": {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys Limited",
        "HDFCBANK.NS": "HDFC Bank",
        "WIPRO.NS": "Wipro Limited",
        "ITC.NS": "ITC Limited",
        "SBIN.NS": "State Bank of India"
    }
}


def test_api_connections():
    status = {
        'yfinance': {'available': YFINANCE_AVAILABLE, 'working': False, 'message': ""},
        'alpha_vantage': {'available': True, 'working': False, 'message': ""}
    }
    
    if YFINANCE_AVAILABLE:
        try:
            test_stock = yf.Ticker("AAPL")
            test_data = test_stock.history(period="5d")
            if not test_data.empty:
                status['yfinance']['working'] = True
                status['yfinance']['message'] = "✅ yfinance connection successful"
            else:
                status['yfinance']['message'] = "❌ yfinance returned no data"
        except Exception as e:
            status['yfinance']['message'] = f"❌ yfinance error: {str(e)[:200]}"
    else:
        status['yfinance']['message'] = "❌ yfinance not installed"
    
    try:
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'AAPL',
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'compact'
        }
        response = requests.get(AV_BASE_URL, params=params, timeout=15)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            status['alpha_vantage']['working'] = True
            status['alpha_vantage']['message'] = "✅ Alpha Vantage connection successful"
        elif 'Note' in data or 'Information' in data:
            status['alpha_vantage']['message'] = "⚠️ Alpha Vantage rate limit exceeded"
        elif 'Error Message' in data:
            status['alpha_vantage']['message'] = f"❌ Alpha Vantage error: {data['Error Message']}"
        else:
            status['alpha_vantage']['message'] = "❌ Unknown Alpha Vantage response"
    except Exception as e:
        status['alpha_vantage']['message'] = f"❌ Alpha Vantage connection failed: {str(e)[:200]}"
    
    return status

@st.cache_data(ttl=300)
def fetch_stock_data_unified(ticker, period="1y"):
    """
    Fetches stock data with a fallback mechanism: yfinance -> Alpha Vantage -> Sample Data.
    """
    # Try yfinance first (if available)
    if YFINANCE_AVAILABLE:
        try:
            stock = yf.Ticker(ticker)
            period_map_yf = {'3mo': '3mo', '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y'}
            history_period = period_map_yf.get(period, '1y')
            df = stock.history(period=history_period)
            
            if not df.empty:
                df = df.reset_index()
                if 'Datetime' in df.columns:
                    df.rename(columns={'Datetime': 'Date'}, inplace=True)
                if 'Date' not in df.columns and df.index.name is not None:
                    df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df.attrs['source'] = 'yfinance'
                return df
        except Exception:
            # let fallback continue
            pass
    
    # Alpha Vantage fallback
    try:
        api_ticker = ticker.replace('.NS', '') if ticker.endswith('.NS') else ticker
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': api_ticker,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full'
        }
        response = requests.get(AV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'Note' in data or 'Information' in data:
            raise Exception("API call frequency limit reached.")
        if 'Error Message' in data:
            raise Exception(data['Error Message'])
        if 'Time Series (Daily)' not in data:
            raise Exception("No data in API response.")
            
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '6. volume': 'Volume'}, inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).sort_index().reset_index().rename(columns={'index': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        
        days = get_period_days(period)
        start_date = datetime.now() - timedelta(days=days)
        df = df[df['Date'] >= start_date]
        
        df.attrs['source'] = 'alpha_vantage'
        return df
        
    except Exception:
        # Final fallback: generated sample data
        return create_sample_data(ticker, period)


def get_period_days(period):
    period_map = {
        '1mo': 30, '3mo': 90, '6mo': 180,
        '1y': 365, '2y': 730, '5y': 1825
    }
    return period_map.get(period, 365)


def create_sample_data(ticker, period):
    days = get_period_days(period)
    
    base_prices = {
        'AAPL': 180, 'GOOGL': 140, 'MSFT': 330, 'TSLA': 250,
        'AMZN': 140, 'META': 300, 'NVDA': 450, 'NFLX': 400,
        'RELIANCE': 2500, 'TCS': 3500, 'INFY': 1500, 'HDFCBANK': 1600
    }
    
    base_name = ticker.split('.')[0].upper()
    base_price = base_prices.get(base_name, 1000)
    
    np.random.seed(abs(hash(ticker)) % 2**32)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    daily_return = 0.08 / 252
    volatility = 0.02
    returns = np.random.normal(daily_return, volatility, days)
    
    prices = [base_price]
    for i in range(1, days):
        new_price = prices[-1] * (1 + returns[i])
        new_price = max(new_price, base_price * 0.5)
        new_price = min(new_price, base_price * 3.0)
        prices.append(new_price)
    
    data = []
    for i, close_price in enumerate(prices):
        daily_vol = abs(np.random.normal(0, 0.015))
        
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        intraday_range = abs(np.random.normal(0, daily_vol))
        high = max(open_price, close_price) * (1 + intraday_range)
        low = min(open_price, close_price) * (1 - intraday_range)
        
        high = max(open_price, close_price, high)
        low = min(open_price, close_price, low)
        
        base_volume = 1000000 if base_price < 1000 else 100000
        volume = int(np.random.lognormal(np.log(base_volume), 0.8))

        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.attrs = {'source': 'sample_data', 'ticker': ticker}
    return df


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def process_stock_data(df, ticker, source):
    if df is None or df.empty:
        return None
    df = df.set_index('Date').sort_index().reset_index()
    
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    for i in [1, 3, 5, 10]:
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    df = df.dropna()
    if df.empty:
        return None
    df.attrs = {'source': source, 'ticker': ticker}
    return df


def prepare_features(df):
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI']
    feature_columns.extend([col for col in df.columns if 'Close_Lag_' in col])
    
    X = df[feature_columns].copy()
    y = df['Close'].copy()
    return X, y, feature_columns


def train_model(df):
    try:
        X, y, feature_names = prepare_features(df)
        if len(df) < 50:
            st.error("Dataset too small for reliable model training.")
            return None, None, None, None
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        y_test_pred = model.predict(X_test_scaled)
        metrics = {
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        return model, scaler, metrics, feature_importance
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None, None, None


def predict_next_price(model, scaler, df):
    try:
        X, _, _ = prepare_features(df)
        if X.empty: return None
        last_features_scaled = scaler.transform(X.iloc[-1:])
        return float(model.predict(last_features_scaled)[0])
    except Exception:
        return None


def get_stock_info(ticker):
    stock_info = {
        'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'currency': 'USD'},
        'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Services', 'currency': 'USD'},
        'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software', 'currency': 'USD'},
        'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'USD'},
        'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'industry': 'Semiconductors', 'currency': 'USD'},
        'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology', 'industry': 'Social Media', 'currency': 'USD'},
        'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical', 'industry': 'E-commerce', 'currency': 'USD'},
        'NFLX': {'name': 'Netflix Inc.', 'sector': 'Communication Services', 'industry': 'Entertainment', 'currency': 'USD'},
        'RELIANCE': {'name': 'Reliance Industries', 'sector': 'Energy', 'industry': 'Oil & Gas', 'currency': 'INR'},
        'TCS': {'name': 'Tata Consultancy Services', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
        'INFY': {'name': 'Infosys Limited', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
        'HDFCBANK': {'name': 'HDFC Bank', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
    }
    
    base_ticker = ticker.split('.')[0].upper()
    info = stock_info.get(base_ticker, {
        'name': ticker,
        'sector': 'Unknown',
        'industry': 'Unknown',
        'currency': 'USD'
    })
    
    info['market_cap'] = 'N/A'
    return info


def display_welcome_page():
    """Displays the welcome page with an overview of the app's features."""
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Hero banner image (use_container_width is the modern parameter)
    st.image(
        "https://placehold.co/1200x300/F0F2F6/007BFF?text=TrendPredict+AI",
        use_container_width=True
    )

    # Welcome text
    st.markdown("""
    ### 👋 Welcome to **TrendPredict AI**

    An advanced financial analysis tool powered by machine learning.  
    Use this app to analyze **market patterns**, predict **price movements**, and gain **deeper insights** into global financial markets.
    """)

    st.markdown("---", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Show a small note if yfinance isn't available locally
    if not YFINANCE_AVAILABLE:
        st.warning("⚠️ `yfinance` is not installed in this environment. The app will attempt Alpha Vantage or use sample data.")

    st.subheader("Core Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>🤖 AI-Powered Engine</h3><p>Our Random Forest model analyzes complex market patterns with precision and intelligence.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🌍 Global Market Access</h3><p>Comprehensive coverage of US and Indian stock markets, plus support for custom tickers.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>📊 Premium Analytics</h3><p>Interactive visualizations, technical indicators, and in-depth model performance analysis.</p></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding-top: 0.5rem">', unsafe_allow_html=True)
    st.subheader("🎯 Platform Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🔬 Technical Features
        - **🧠 Machine Learning Model**: Random Forest with advanced feature engineering.
        - **📈 Technical Indicators**: RSI, Moving Averages, and Volume analysis.
        - **🎨 Interactive Charts**: Candlestick, Volume, and Momentum visualizations.
        - **⚡ Real-time Processing**: Live data feeds with intelligent caching.
        - **🔍 Model Validation**: Robust testing for performance evaluation.
        """)
    with c2:
        st.markdown("""
        #### 🌐 Market Coverage
        - **🇺🇸 US Technology & Finance**: Apple, Google, NVIDIA, JPMorgan, and more.
        - **🇮🇳 Indian Giants**: Reliance, TCS, Infosys, HDFC Bank, and others.
        - **🔧 Custom Tickers**: Full support for any valid stock symbol.
        - **⏱️ Flexible Periods**: From 3 months to 5 years of historical data.
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding-top: 0.5rem">', unsafe_allow_html=True)
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    1.  **🎛️ Configure**: Select your target market and stock using the sidebar. For custom stocks, choose 'Custom Ticker'.
    2.  **⏳ Select Period**: Use the slider to choose the historical analysis period.
    3.  **🤖 Engage AI**: Click the **'Run Analysis'** button to activate the prediction engine.
    4.  **📊 Explore Insights**: Navigate through the dashboard tabs to view predictions, charts, and model analytics.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="warning-card">
    <strong>Disclaimer:</strong> This tool is for educational and informational purposes only and does not constitute financial advice. All predictions are based on historical data and do not guarantee future results.
    </div>""", unsafe_allow_html=True)


def display_analysis_dashboard():
    """Displays the main analysis dashboard."""
    ticker = st.session_state.ticker
    period = st.session_state.period

    with st.spinner(f"Initiating analysis for {ticker}..."):
        df_raw = fetch_stock_data_unified(ticker, period=period)

    if df_raw is None or df_raw.empty:
        st.error("Data acquisition failed from all sources. Please verify the ticker or try again.")
        return
    else:
        source = df_raw.attrs.get('source', 'unknown')
        st.success(f"Fetched data from {source.replace('_', ' ').title()}.")
        with st.spinner("Processing temporal data and training AI core..."):
            df = process_stock_data(df_raw, ticker, source)
        
        if df is None or df.empty:
            st.error("Data processing anomaly. Insufficient data for AI training after feature engineering.")
            return

        stock_info = get_stock_info(ticker)
        st.title(f"📈 {stock_info['name']} ({ticker}) Intelligence Dashboard")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🧠 AI Prediction", "📊 Technical Charts", "⚙️ Model DNA"])

        with tab1:
            st.header("Market Snapshot")
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2]) if len(df) >= 2 else current_price
            price_change = current_price - prev_price
            pct_change = (price_change / prev_price) * 100 if prev_price != 0 else 0
            currency_symbol = '₹' if stock_info['currency'] == 'INR' else '$'

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cls = 'positive' if price_change >= 0 else 'negative'
                st.markdown(f"<div class=\"metric-card\"><h3>Last Close</h3><p>{currency_symbol}{current_price:.2f}</p><span class=\"{cls}\">{price_change:+.2f} ({pct_change:.2f}%)</span></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class=\"metric-card\"><h3>Day Range</h3><p>{currency_symbol}{df['Low'].iloc[-1]:.2f} - {currency_symbol}{df['High'].iloc[-1]:.2f}</p><span>Today's High-Low</span></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class=\"metric-card\"><h3>Volume</h3><p>{df['Volume'].iloc[-1]/1e6:.2f}M</p><span>Shares Traded</span></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div class=\"metric-card\"><h3>52-Wk High</h3><p>{currency_symbol}{df['High'].max():.2f}</p><span>Annual Peak</span></div>", unsafe_allow_html=True)

            fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_20'], mode='lines', name='20-Day MA', line=dict(width=2)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'], mode='lines', name='50-Day MA', line=dict(width=2)))
            fig.update_layout(title='Price Action with Moving Averages', template='plotly_white', xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True, key="main_chart")

        with tab2:
            st.header("AI Prediction Engine")
            with st.spinner("Calibrating predictive model..."):
                model, scaler, metrics, feature_importance = train_model(df)

            if model is not None:
                prediction = predict_next_price(model, scaler, df)
                if prediction is not None:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    pred_cols = st.columns((2, 2, 3))
                    with pred_cols[0]:
                        st.metric("Last Closing Price", f"{currency_symbol}{current_price:.2f}")
                    with pred_cols[1]:
                        st.metric("AI Predicted Next Close", f"{currency_symbol}{prediction:.2f}")

                    pred_change = prediction - current_price
                    pred_pct = (pred_change / current_price) * 100 if current_price != 0 else 0
                    with pred_cols[2]:
                        st.metric("Anticipated Movement", f"{pred_pct:.2f}%", f"{pred_change:+.2f} {stock_info['currency']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("AI Core generated no prediction.")
            else:
                st.error("AI Core calibration failed. Unable to generate prediction.")

        with tab3:
            st.header("Advanced Technical Charts")
            c1, c2 = st.columns(2)
            with c1:
                fig_vol = px.bar(df, x='Date', y='Volume', title='Trading Volume Analysis')
                fig_vol.update_layout(template='plotly_white')
                st.plotly_chart(fig_vol, use_container_width=True, key="volume_chart")
            with c2:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold")
                fig_rsi.update_layout(title='Relative Strength Index (RSI)', template='plotly_white', yaxis_range=[0,100])
                st.plotly_chart(fig_rsi, use_container_width=True, key="rsi_chart")

        with tab4:
            st.header("Model DNA: Deconstructing the AI")
            if metrics:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Model Performance Evaluation")
                perf_cols = st.columns(2)
                perf_cols[0].metric("Predictive Accuracy (R² Score)", f"{metrics['test_r2']:.2%}")
                perf_cols[1].metric("Avg. Prediction Error (MAE)", f"{currency_symbol}{metrics['test_mae']:.2f}")
                st.info(f"The model's predictions on unseen test data were, on average, off by {currency_symbol}{metrics['test_mae']:.2f}. The R² score indicates how much of the price variation the model could explain.")
                st.markdown("</div>", unsafe_allow_html=True)

            if feature_importance is not None:
                st.markdown('<div class="card" style="margin-top: 2rem;">', unsafe_allow_html=True)
                st.subheader("Feature Influence Matrix")
                st.write("This chart shows which data points the AI considered most important when making its predictions. Higher values indicate greater influence.")
                fig_imp = px.bar(feature_importance.head(10), x='importance', y='feature', orientation='h', title="Top 10 Most Influential Features")
                fig_imp.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True, key="feature_importance_chart")
                st.markdown("</div>", unsafe_allow_html=True)


def main():
    """
    Main function to run the Streamlit application.
    It handles the overall layout and flow.
    """
    # Initialize session state variables if they don't exist
    if 'data_source_success' not in st.session_state:
        st.session_state.data_source_success = None

    if 'ticker' not in st.session_state:
        st.session_state.ticker = "AAPL"
    if 'period' not in st.session_state:
        st.session_state.period = "1y"
    if 'analyze_button_clicked' not in st.session_state:
        st.session_state.analyze_button_clicked = False

    def set_analyze_button_clicked():
        st.session_state.analyze_button_clicked = True

    with st.sidebar:
        st.markdown("<h3>TrendPredict AI 📈</h3>", unsafe_allow_html=True)
        st.markdown("AI-Powered Market Intelligence")
        st.markdown("---")

        st.subheader("Analysis Configuration")
        
        market_categories = list(MARKET_STOCKS.keys()) + ["⚙️ Custom Ticker"]
        market_category = st.selectbox("Market Sector", market_categories, key='market_category')

        if market_category == "⚙️ Custom Ticker":
            st.text_input(
                "Enter Custom Ticker",
                value=st.session_state.ticker,
                placeholder="e.g., AAPL, RELIANCE.NS",
                key='ticker'
            )
        else:
            stock_options = MARKET_STOCKS[market_category]
            options = list(stock_options.keys())
            
            default_ticker = st.session_state.ticker
            if default_ticker not in options:
                default_ticker = options[0]  # fallback to first option
                st.session_state.ticker = default_ticker

            st.selectbox(
                "Select Stock",
                options,
                index=options.index(default_ticker),
                format_func=lambda x: f"{x} - {stock_options[x]}",
                key='ticker'
            )

        st.select_slider(
            "Analysis Period",
            options=["3mo", "6mo", "1y", "2y", "5y"],
            value=st.session_state.period,
            key='period'
        )
        
        st.markdown("---")
        st.button("🚀 Run Analysis", use_container_width=True, on_click=set_analyze_button_clicked)
        st.markdown("---")
        
        st.markdown("### 🔧 System Status")
        if st.button("🔍 Check API Health", use_container_width=True):
            with st.spinner("Running diagnostics..."):
                api_status = test_api_connections()
            
            if api_status['alpha_vantage']['working']:
                st.markdown('<div class="status-success">🟢 Alpha Vantage Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error">🔴 Alpha Vantage Offline</div>', unsafe_allow_html=True)
                
            if api_status['yfinance']['working']:
                st.markdown('<div class="status-success">🟢 yfinance Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-warning">🟡 yfinance Unavailable</div>', unsafe_allow_html=True)

    if st.session_state.analyze_button_clicked:
        display_analysis_dashboard()
    else:
        display_welcome_page()


if __name__ == "__main__":
    main()
