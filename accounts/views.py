from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required
from .models import AngelClient
from .utils import AngleLogin

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            auth_login(request, user)
            if AngelClient.objects.filter(app_user__user=user).exists():
                return redirect('success')
            else:
                return render(request, 'accounts/login.html', {'error': 'User is not associated with any client.'})
        else:
            return render(request, 'accounts/login.html', {'error': 'Invalid username or password.'})
    
    return render(request, 'accounts/login.html')



@login_required
def success(request):
    user = request.user
    try:
        angel_client = AngelClient.objects.get(app_user__user=user)
        client_id = angel_client.client_id
        app_passcode = angel_client.app_passcode
        api_key = angel_client.api_key
        access_token = angel_client.access_token
        client = AngleLogin(client_id=client_id,app_password=app_passcode,api_key=api_key,access_token=access_token)
        client_balance = client['smartapi'].rmsLimit()
        context = {
            'client_id': client_id,
            'client_name':client['profile']['name'],
            'client_balance':client_balance['data']
            
            }



    except angel_client.DoesNotExist:
        client_id = None

            


  
    return render(request, 'accounts/success.html', context)
