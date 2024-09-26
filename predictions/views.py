from django.shortcuts import render, redirect
from django.http import HttpResponse
import logging
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
import datetime
import os
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import CSVUploadForm
from .models import StockPrediction,Option,Order
from .utils import get_data,get_token_for_angel,get_live_data,place_order_angel,get_upcoming_last_thursdays,get_1_min_candle_data,tickers,process_ticker
from accounts.utils import AngleLogin
from accounts.models import AngelClient
from django.shortcuts import render, get_object_or_404
import time



def index(request):
    return render(request, 'accounts/login.html')
@login_required
def home(request):
    if request.user is not None:
        upcoming_thursdays = get_upcoming_last_thursdays()
        user = request.user
        angel_client = AngelClient.objects.get(app_user__user=user)
        client_id = angel_client.client_id
        app_passcode = angel_client.app_passcode
        api_key = angel_client.api_key
        access_token = angel_client.access_token
        client = AngleLogin(client_id=client_id, app_password=app_passcode, api_key=api_key, access_token=access_token)
        client_obj = client['smartapi']

        rms_data = client_obj.rmsLimit()
        try:
            available_cash = rms_data['data']['availablecash']
            if request.method == 'POST':
                if 'use_all' in request.POST:
                    amount = available_cash
                else:
                    amount = request.POST.get('manual_amount')
                return render(request, 'predictions/home.html', {'available_cash': available_cash, 'selected_amount': amount,'thursdays': upcoming_thursdays,'available_cash':available_cash})
        except Exception as e:
            print('this is execption ',e)
        
        return render(request, 'predictions/home.html',{'thursdays': upcoming_thursdays,'available_cash':available_cash})
    else:
        return render(request,'accounts/login.html')
@login_required
def predict_stocks(request):
    stock_count = []
    user = request.user
    angel_client = AngelClient.objects.get(app_user__user=user)
    client_id = angel_client.client_id
    app_passcode = angel_client.app_passcode
    api_key = angel_client.api_key
    access_token = angel_client.access_token
    client = AngleLogin(client_id=client_id, app_password=app_passcode, api_key=api_key, access_token=access_token)
    if request.method == 'POST':
        results = []
        expiry = request.POST.get('expiry')
        with ThreadPoolExecutor(max_workers=196) as executor:
            futures = [executor.submit(process_ticker, ticker) for ticker in tickers]
            for index, future in enumerate(futures):  # Use enumerate to get the index
                results.append(future.result())
            
                # Calculate the progress percentage
        for result in results:
            if (result['movement'] >= 0.40 and result['close_5_min'] > result['previous_5_min_close']) or (result['movement'] <= -0.40 and result['close_5_min'] < result['previous_5_min_close']):
                stock_count.append(result)
    rms_limit = client['smartapi'].rmsLimit()
    available_cash = rms_limit['data']['availablecash']
    if len(stock_count)>1:
        per_stock_capital = float(available_cash)/len(stock_count)
    else:
        per_stock_capital = float(available_cash)/2
    for stock in stock_count:
        option_data = get_token_for_angel(
            ticker=stock['ticker'],
            expiry=expiry,
            current_price=stock['last_price'],
            movement=stock['movement'],
            )
        token = option_data['instrument_token']
        symbol = option_data['instrument_symbol']
        lotsize = option_data['lotsize']
        print(stock)
        try:
            ltp1 = get_1_min_candle_data(security_id=token,exchange='NFO',api_obj=client)['close']
            ltp2 = get_live_data(exchange='NFO', token=token, mode='LTP', client=client)
            ltp = min(ltp1,ltp2)
        except:
            ltp = get_live_data(exchange='NFO', token=token, mode='LTP', client=client)
        per_lot_value = ltp * lotsize
        multiplier  = per_stock_capital // per_lot_value
        lotsize *= multiplier
        orderid = place_order_angel(token=token, symbol=symbol, lotsize=lotsize, OrderPrice=ltp, client=client)
        #orderid = None
        stock_prediction = StockPrediction.objects.create(
            user = request.user,
            ticker=stock['ticker'],
            last_price=stock['last_price'],
            predicted_price=stock['predicted_price'],
            target_price=stock['target_price'],
            movement=round(stock['movement'], 2),
            volume=stock['volume'],
            volume_change=stock['volume_change'],
            timestamp=datetime.datetime.now()
            )
                
        if option_data:
            option = Option.objects.create(
                stock_prediction=stock_prediction,
                option_strike=option_data['instrument_symbol'],
                option_type='c/p',
                security_id=option_data['instrument_token'],
                lot_size=option_data['lotsize']
                )
            if orderid is not None:
                Order.objects.create(
                    option = option,
                    order_id = orderid,
                    option_strike = symbol,
                    price = ltp,
                    lot_size = lotsize
                )
            else:
                Order.objects.create(
                    option = option,
                    order_id = None,
                    option_strike = symbol,
                    price = ltp,
                    lot_size = lotsize
                )



        
    return redirect('results')

        
                                        


 

        

    

def results(request):
    if request.user.is_authenticated:
        predictions = StockPrediction.objects.filter(user=request.user).order_by('-timestamp')
        orders = Order.objects.filter(option__stock_prediction__user=request.user).order_by('-order_time')
        return render(request, 'predictions/results.html', {'predictions': predictions,'orders':orders})
    else:
        return redirect('login')


def set_loss(request,order_id):
    if request.method =='POST':

        order = get_object_or_404(Order,order_id=order_id)
        loss  = (order.price - order.stoploss)*order.lot_size
        order.loss = 0-loss
        order.profit = 0
        order.save()
        return redirect('results')
def set_profit(request,order_id):
    if request.method =='POST':

        order = get_object_or_404(Order,order_id=order_id)
        profit = (order.target - order.price)*order.lot_size
        order.profit = profit
        order.loss = 0
        order.save()
        return redirect('results')


def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['csv_file']

            if settings.STATICFILES_DIRS:
                data_dir = os.path.join(settings.STATICFILES_DIRS[0], 'data')
                os.makedirs(data_dir, exist_ok=True)  # Ensure the directory exists
                fs = FileSystemStorage(location=data_dir)
                filename = fs.save(csv_file.name, csv_file)
                file_path = os.path.join(data_dir, filename)

                # Process the CSV file
                df = pd.read_csv(file_path)

                # Split "SEM_TRADING_SYMBOL" into "TICKER" and "EXPIRY"
                df[['TICKER', 'EXPIRY']] = df['SEM_TRADING_SYMBOL'].str.extract(r'([A-Z]+)-(\w+)')
                df = df[df['EXPIRY'].str.contains('Aug2024', na=False)]
                df = df[df['SEM_INSTRUMENT_NAME'].str.contains('OPTSTK', na=False)]

                # Save the updated CSV file
                updated_filename = f"updated_{csv_file.name}"
                updated_file_path = os.path.join(data_dir, updated_filename)
                df.to_csv(updated_file_path, index=False)

                file_url = fs.url(updated_filename)
                return render(request, 'data/upload_success.html', {'file_url': file_url})
            else:
                return render(request, 'data/upload_csv.html', {
                    'form': form,
                    'error': 'Static files directory is not configured.'
                })
    else:
        form = CSVUploadForm()
    
    return render(request, 'data/upload_csv.html', {'form': form})