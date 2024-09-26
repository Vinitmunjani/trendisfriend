#Angle login
from SmartApi import SmartConnect #or from SmartApi.smartConnect import SmartConnect
import pyotp
from logzero import logger
from .models import AngelClient
#from dhanhq import dhanhq

def AngleLogin(client_id,app_password,api_key,access_token):

    smartApi = SmartConnect(api_key)
    try:
        access_token  = access_token
        #token generated from totp smartapi
        totp = pyotp.TOTP(access_token).now()
    except Exception as e:
        logger.error("Invalid Token: The provided token is not valid.")
        raise e
    correlation_id = "abcde"
    data = smartApi.generateSession(client_id,app_password, totp)
    refreshToken = data['data']['refreshToken']
    profile = smartApi.getProfile(refreshToken)['data']
    
    if data['status'] == False:
        logger.error(data)
    else:
        return {
            'smartapi':smartApi,
            'profile':profile
        }
    """
    # login api call
    # logger.info(f"You Credentials: {data}")
        authToken = data['data']['jwtToken']
        refreshToken = data['data']['refreshToken']
    # fetch the feedtoken
        feedToken = smartApi.getfeedToken()
    
    # fetch User Profile
        profile = smartApi.getProfile(refreshToken)
        funds = smartApi.rmsLimit()
        print(funds)
        smartApi.generateToken(refreshToken)

        res=profile['data']['name']
        return res,funds['data']
    """


"""

def DhanLogin(client_id,access_token):
    dhan = dhanhq(client_id,access_token)
    return dhan

"""