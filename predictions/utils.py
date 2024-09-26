import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import yfinance as yf
import ta
import os
from django.conf import settings
from SmartApi import SmartConnect #or from SmartApi.smartConnect import SmartConnect
import pyotp
from logzero import logger
import calendar
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

tickers = [ 'BAJAJFINSV.NS','TECHM.NS','HDFCBANK.NS', 'RELIANCE.NS', 'INFY.NS',
             'AXISBANK.NS', 'BAJFINANCE.NS', 'TITAN.NS',
               'INDUSINDBK.NS', 'POWERGRID.NS',  'JSWSTEEL.NS',
                 'HDFCLIFE.NS', 'TATACONSUM.NS', 'LTIM.NS', 'DIVISLAB.NS',
                   'UPL.NS', 'CIPLA.NS', 'GRASIM.NS', 'TATAMOTORS.NS', 'BRITANNIA.NS',
                     'NTPC.NS', 'DRREDDY.NS', 'ICICIBANK.NS', 'NESTLEIND.NS', 
                     'APOLLOHOSP.NS', 'SUNPHARMA.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'MARUTI.NS',
                       'ADANIPORTS.NS', 'ASIANPAINT.NS', 'WIPRO.NS', 'KOTAKBANK.NS', 'M&M.NS',
                         'HINDALCO.NS', 'TCS.NS', 'SBILIFE.NS', 'ITC.NS', 
                         'HCLTECH.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS','ONGC.NS']



def get_data(ticker,interval,method,days=None):

    today = datetime.now()
    if interval == '1m':
        start_date = today -timedelta(days=7)
        if method == 'test':
            end_date = today - timedelta(days=1)
        elif method == 'train':
            end_date = today
    elif interval == '15m':
        start_date = today- timedelta(days=58)
        if method == 'test' or method == 'test_today':
            end_date = today 
        elif method == 'train':
            end_date = today
    elif interval == '5m' and method == 'test':
        start_date = today - timedelta(days=59)
        end_date = today          
    stock = yf.Ticker(ticker)
    dataset = stock.history(period="59d", interval=interval, prepost=True, auto_adjust=True)   
    #dataset = yf.download(ticker,start=start_date,end=end_date,interval=interval)
    #Perform the desired data transformations and feature engineering
    dataset["MA"] = ta.trend.sma_indicator(dataset["Close"], window=10)
    dataset["ma50"] = ta.trend.sma_indicator(dataset["Close"], window=50)
    dataset['prev_close'] = dataset['Close'].shift(1)
    dataset['prev_high'] = dataset['High'].shift(1)
    dataset['prev_low'] = dataset['Low'].shift(1)
    dataset['prev_open'] = dataset['Open'].shift(1)
    dataset['Moving Average (200)'] = dataset['Close'].rolling(window=200).mean()
    dataset['Price Rate of Change'] = dataset['Close'].pct_change(periods=1)
    dataset['Volume'] = dataset['Volume'].replace(0,np.nan)
    dataset['Volume'] = dataset['Volume'].fillna(method='ffill')
    dataset['Volume Rate of Change'] = dataset['Volume'].pct_change(periods=1)
    dataset['midpoint'] = (dataset['High'] + dataset['Low']) / 2
    dataset['pred_high'] = dataset['High'].shift(-1)
    dataset['pred_low'] = dataset['Low'].shift(-1)
    dataset['pred_close'] = dataset['Close'].shift(-1)
    dataset['pred_open'] = dataset['Open'].shift(-1)
    bollinger_bands = ta.volatility.BollingerBands(dataset['Close'])
    dataset['Volatility'] = dataset['High'] - dataset['Low']
    dataset['BB_upper'] = bollinger_bands.bollinger_hband()
    dataset['BB_lower'] = bollinger_bands.bollinger_lband()
    dataset['VWAP'] =(dataset['Volume'] * (dataset['High'] + dataset['Low'] + dataset['Close']) / 3).cumsum() / dataset['Volume'].cumsum()
    dataset['vwap_pr_change'] = dataset['VWAP'].pct_change(periods=1)
    dataset['bb_u_pr_actual'] = (dataset['BB_upper'] - dataset['Close']) * 100 / dataset['Close']
    dataset['bb_l_pr_actual'] = (dataset['BB_lower'] - dataset['Close']) * 100 / dataset['Close']
    dataset['bb_uper_pr_change'] = dataset['BB_upper'].pct_change(periods=1)
    dataset['BB_lower_pr_change'] = dataset['BB_lower'].pct_change(periods=1)
    dataset['Volume Rate of Change'] = dataset['Volume'].pct_change(periods=1)
    dataset['volatility_pr_change'] = dataset['Volatility'].pct_change(periods=1)
    dataset['High_Low_ratio'] = dataset['High'] / dataset['Low']
    dataset['ma200_pct_to_actual'] = (dataset['Moving Average (200)'] - dataset['Close']) * 100 / dataset['Close']
    dataset['ma50_pct_to_actual'] = (dataset['ma50'] - dataset['Close']) * 100 / dataset['Close']
    dataset['ma10_pct_to_actual'] = (dataset['MA'] - dataset['Close']) * 100 / dataset['Close']
    dataset["RSI"] = ta.momentum.rsi(dataset["Close"])
    dataset['pr_change_close'] = (dataset['Close'] - dataset['prev_close']) * 100 / dataset['prev_close']
    dataset['pr_change_high'] = (dataset['High'] - dataset['prev_high']) * 100 / dataset['prev_high']
    dataset['pr_change_low'] = (dataset['Low'] - dataset['prev_low']) * 100 / dataset['prev_low']
    dataset['pr_change_open'] = (dataset['Open'] - dataset['prev_open']) * 100 / dataset['prev_open']
    dataset['pred_high_pr'] = (dataset['pred_high'] - dataset['High']) * 100 / dataset['High']
    dataset['pred_open_pr'] = (dataset['pred_open'] - dataset['Open']) * 100 / dataset['Open']
    dataset['stho'] = ta.momentum.StochasticOscillator(dataset['High'], dataset['Low'], dataset['Close'], window=14).stoch()
    """
    Q1 = dataset['pr_change_close'].quantile(0.25)
    Q3 = dataset['pr_change_close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = ((dataset[['pr_change_close']] < lower_bound) | (dataset[['pr_change_close']] > upper_bound))
    dataset[['pr_change_close']][outlier_indices] = np.nan
    dataset.replace([np.inf, -np.inf], np.nan)
    """
    
    data = dataset
    if method == 'test' and  interval == '1m':
        return data[len(data)-374:]
    if method == 'train' and  interval == '1m':
        return data
    if method == 'test' and  interval == '15m':
        dataset =  data[len(data)-(days*25):(len(data)-(days*25))+27]
        data = dataset.copy()
        data.fillna(0,inplace=True)
        return (data,data.iloc[-1,3])
    if interval == '5m' and method == 'test':
        dataset =  data[len(data)-(days*75):(len(data)-(days*75))+74]
        
        data = dataset.copy()
        data.replace([np.inf, -np.inf], np.nan)
        data.fillna(0,inplace=True)
        return (data,data.iloc[-1,3])
    if method == 'test' and interval == '1d':
        data = dataset.copy()
        return  (data.dropna(inplace=True),data.iloc[-1,3])
    if method == 'train' and interval == '15m':
        return data
    if method == 'test_today' and interval == '15m':
        return data[:len(data)-25]


model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model11.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist.")

model = load_model(model_path)
def process_ticker(ticker):
    i =1
    new_data = get_data(ticker, '15m', 'test', i)
    data_5_min = get_data(ticker,'5m','test',i)[0]
    dataset = new_data[0]
    close = new_data[1]
    volume_change = dataset['Volume Rate of Change'][-1]
    target_column = 'pred_high_pr'
    feature_columns = ['bb_u_pr_actual', 'bb_l_pr_actual', 'bb_uper_pr_change', 'BB_lower_pr_change',
                       'Volume Rate of Change', 'vwap_pr_change', 'volatility_pr_change', 'High_Low_ratio', 
                       'ma200_pct_to_actual', 'ma50_pct_to_actual', 'ma10_pct_to_actual', 'RSI', 
                       'pr_change_close', 'pr_change_high', 'pr_change_low', 'pr_change_open']
    
    y_train = dataset[target_column].values
    scaler = MinMaxScaler()
    scaler.fit(dataset[feature_columns])
    new_data_scaled = scaler.transform(new_data[0][feature_columns])
    sequence_length = 5
    expected_shape = (1, sequence_length, len(feature_columns))
    num_samples = new_data_scaled.shape[0]
    if num_samples < sequence_length:
        raise ValueError(f"Insufficient data for prediction. At least {sequence_length} samples are required for ticker {ticker}.")
    start_index = num_samples - sequence_length
    reshaped_data = np.reshape(new_data_scaled[start_index:], expected_shape)
    predicted_values = model.predict(reshaped_data)
    scaled_predictions = predicted_values * (np.max(y_train) - np.min(y_train)) + np.min(y_train)
    close = round(close, 2)
    stock_name = ticker.split(".")[0]
    value = round(close + ((scaled_predictions[0][0] * close) / 100), 2)
    movement = scaled_predictions[0][0]
    close_5_min = data_5_min['Close'][-1]
    previous_5_min_close = data_5_min['Close'][-2]

    return {
        'ticker': stock_name,
        'last_price': close,
        'predicted_price': value,
        'target_price': value,
        'movement': movement,
        'volume':dataset['Volume'][-1],
        'volume_change':volume_change,
        'close_5_min':close_5_min,
        'previous_5_min_close':previous_5_min_close
        


    
    }


#for geting token of the script
def get_token_for_angel(ticker, expiry=None, current_price=None, movement=None):
    instrument_df = pd.read_csv('static/data/api_script_master.csv')
    if expiry and current_price and movement:
        current_price *= 100
        option_type = 'PE' if movement < 0 else 'CE'
        instrument_df = instrument_df[
            (instrument_df['name'] == ticker) &
            (instrument_df['expiry'] == expiry) &
            (instrument_df['exch_seg'] == 'NFO') &
            (instrument_df['symbol'].str.endswith(option_type))
        ]
        
        instrument_df['strike_diff'] = abs(instrument_df['strike'] - current_price)
        nearest_strike = instrument_df.loc[instrument_df['strike_diff'].idxmin()]['strike']
        instrument_df = instrument_df[instrument_df['strike'] == nearest_strike]
        # Convert Series to scalar values
        dict_values = {
            "instrument_token": int(instrument_df['token'].values[0]),
            "instrument_symbol": instrument_df['symbol'].values[0],
            "lotsize": int(instrument_df['lotsize'].values[0])
        }
    elif ticker:
        instrument_df = instrument_df[instrument_df['name'] == ticker]
        instrument_df = instrument_df[instrument_df['exch_seg']=='NSE']
        instrument_df = instrument_df[instrument_df['symbol'].str.endswith('EQ')]
        
        dict_values = {
            "instrument_token": instrument_df['token'],
            "instrument_symbol": instrument_df['symbol']
        }

    
    return dict_values


#for the setting sl and target with round number , multiplier of 0.05
def round_to_nearest_multiple(number, multiple=0.05):
    rounded_down = (number // multiple) * multiple
    rounded_up = rounded_down + multiple
    # Check which is closer, rounded_down or rounded_up
    if number - rounded_down < rounded_up - number:
        return round(rounded_down, 2)
    else:
        return round(rounded_up, 2)
    

#gets the last traded price of ticker
def get_live_data(exchange,token,mode,client):
    
    if mode == 'LTP':   
        live_data = client['smartapi'].getMarketData(mode='LTP',exchangeTokens={exchange:[token]})
        ltp = live_data['data']['fetched'][0]['ltp']
        return float(ltp)
    elif mode == 'OHLC':
        live_data = client['smartapi'].getMarketData(mode='OHLC',exchangeTokens={exchange:[token]})
        open = live_data['data']['fetched'][0]['open']
        high = live_data['data']['fetched'][0]['high']
        low = live_data['data']['fetched'][0]['low']
        close = live_data['data']['fetched'][0]['close']
        ltp = live_data['data']['fetched'][0]['ltp']

        data = {
            'open':open,
            'high':high,
            'low':low,
            'close':close,
            'ltp':ltp
        }
        return data
    
#ORDER  PLACEMENT
def place_order_angel(token,symbol,lotsize,OrderPrice,client):
    target = OrderPrice / 10
    target = round_to_nearest_multiple(target)
    stoploss = OrderPrice / 20
    stoploss = round_to_nearest_multiple(stoploss)
    
    order_value = OrderPrice*lotsize
    #fund = get_funds()
    #funds = client.rmsLimit()
    #available_limit = float(funds['data']['availablecash'])
    available_limit = 100000
    if order_value < available_limit:
        try :
            orderparams = {
            "variety": "ROBO",
            "tradingsymbol":symbol,
            "symboltoken":str(token),
            "transactiontype": "BUY",
            "exchange": "NFO",
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price":float(OrderPrice),
            "squareoff":float(target),
            "stoploss":float(stoploss),
            "quantity":int(lotsize)
            }
            order = client['smartapi'].placeOrder(orderparams)
        
            return order
        except Exception as e:
            message = e
    else:
        message = 'insufficiant account balance for this order'
    return message

   
#get the all the expiry of thursdays
def get_upcoming_last_thursdays():
    current_date = datetime.now()
    current_year = current_date.year
    # Store the upcoming last Thursdays
    upcoming_last_thursdays = []
    # Loop through the current year and one upcoming year (current and next year)
    for year in [current_year, current_year + 1]:
        for month in range(1, 13):
            # Get the last day of the month
            last_day = calendar.monthrange(year, month)[1]
            
            # Create a datetime object for the last day of the month
            last_date = datetime(year, month, last_day)
            
            # Calculate how many days to subtract to get to the last Thursday
            days_to_thursday = (last_date.weekday() - 3) % 7
            
            # Subtract the days to get the last Thursday
            last_thursday = last_date - timedelta(days=days_to_thursday)
            
            # Add only future dates or today's date
            if last_thursday >= current_date:
                # Format the date in '27JAN2028' format
                formatted_date = last_thursday.strftime('%d%b%Y').upper()
                upcoming_last_thursdays.append(str(formatted_date))
    return upcoming_last_thursdays

# get the last 1 min candle of sript
def get_1_min_candle_data(security_id,exchange,api_obj):
    try:
        now = datetime.now()
        one_min_ago = now - timedelta(minutes=1)
        from_time = one_min_ago.strftime("%Y-%m-%d %H:%M")
        to_time = one_min_ago.strftime("%Y-%m-%d %H:%M")
        params = {
            "exchange": exchange,
            "symboltoken": str(security_id),
            "interval": "ONE_MINUTE",
            "fromdate": from_time,  # Set the from date to today's 9:35 AM
            "todate": to_time     # Set the to date to the same time for 9:35 AM candle
        }
        response = api_obj['smartapi'].getCandleData(params)
        if response['status']:
            candle_data = response['data'][0]
            candle = {
                'open':candle_data[1],
                'high':candle_data[2],
                'low':candle_data[3],
                'close':candle_data[4],
            }
            return candle
    except Exception as e:
        print(f"Error getting 1 min candle data: {e}")

