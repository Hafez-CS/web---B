from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
import requests
from decimal import Decimal


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_dollar(request):
    try:
        response = requests.get('https://api.bonbast.com/api/v1/currencies', timeout=20)
        if response.status_code == 200:
            data = response.json()
            dollar_price = data.get('usd', {}).get('sell', 0)
            return JsonResponse({
                'success': True,
                'currency': 'USD',
                'price': dollar_price,
                'unit': 'ریال',
                'timestamp': response.headers.get('Date')
            })
        
        response = requests.get('https://api.tgju.org/v1/market/indicator/summary-table-data/price_dollar_rl', timeout=30)
        if response.status_code == 200:
            data = response.json()
            dollar_price = data.get('p', 0)
            return JsonResponse({
                'success': True,
                'currency': 'USD',
                'price': dollar_price,
                'unit': 'ریال',
                'timestamp': None
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': 'error'
        }, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_gold(request):
    try:
        response = requests.get('https://api.tgju.org/v1/market/indicator/summary-table-data/geram18', timeout=30)
        if response.status_code == 200:
            data = response.json()
            gold_price = data.get('p', 0)
            return JsonResponse({
                'success': True,
                'currency': 'GOLD',
                'name': 'طلا 18 عیار',
                'price': gold_price,
                'unit': 'ریال',
                'timestamp': None
            })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message':'error'
        }, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_bors(request):
    try:
        response = requests.get('https://api.tgju.org/v1/market/indicator/summary-table-data/shakhes', timeout=30)
        if response.status_code == 200:
            data = response.json()
            bors_index = data.get('p', 0)
            return JsonResponse({
                'success': True,
                'name': 'شاخص کل بورس',
                'index': bors_index,
                'timestamp': None
            })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': 'error'
        }, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_bitcoin(request):
    try:
        btc_response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd', timeout=30)
        
        if btc_response.status_code == 200:
            btc_data = btc_response.json()
            btc_price_usd = btc_data.get('bitcoin', {}).get('usd', 0)
            

            try:
                usd_response = requests.get('https://api.tgju.org/v1/market/indicator/summary-table-data/price_dollar_rl', timeout=30)
                if usd_response.status_code == 200:
                    usd_data = usd_response.json()
                    dollar_price = usd_data.get('p', 0)
                    btc_price_rial = btc_price_usd * dollar_price
                else:
                    dollar_price = 0
                    btc_price_rial = 0
            except:
                dollar_price = 0
                btc_price_rial = 0
            
            return JsonResponse({
                'success': True,
                'currency': 'BTC',
                'name': 'بیت کوین',
                'price_usd': btc_price_usd,
                'price_rial': btc_price_rial,
                'dollar_rate': dollar_price,
                'unit': 'دلار/ریال',
                'timestamp': None
            })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': 'error'
        }, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_currencies(request):
    result = {
        'success': True,
        'data': {}
    }
    
    try:
        try:
            response = requests.get('https://api.tgju.org/v1/market/indicator/summary-table-data/price_dollar_rl', timeout=30)
            if response.status_code == 200:
                data = response.json()
                result['data']['dollar'] = {
                    'price': data.get('p', 0),
                    'change': data.get('d', 0),
                    'unit': 'ریال'
                }
        except:
            result['data']['dollar'] = {'price': 0, 'error': True}
        
        try:
            response = requests.get('https://api.tgju.org/v1/market/indicator/summary-table-data/price_eur', timeout=30)
            if response.status_code == 200:
                data = response.json()
                result['data']['euro'] = {
                    'price': data.get('p', 0),
                    'change': data.get('d', 0),
                    'unit': 'ریال'
                }
        except:
            result['data']['euro'] = {'price': 0, 'error': True}
        
        try:
            response = requests.get('https://api.tgju.org/v1/market/indicator/summary-table-data/geram18', timeout=30)
            if response.status_code == 200:
                data = response.json()
                result['data']['gold'] = {
                    'price': data.get('p', 0),
                    'change': data.get('d', 0),
                    'unit': 'ریال'
                }
        except:
            result['data']['gold'] = {'price': 0, 'error': True}
        
        try:
            response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd', timeout=30)
            if response.status_code == 200:
                data = response.json()
                result['data']['bitcoin'] = {
                    'price': data.get('bitcoin', {}).get('usd', 0),
                    'unit': 'دلار'
                }
        except:
            result['data']['bitcoin'] = {'price': 0, 'error': True}
        
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'message': 'error'
        }, status=500)
