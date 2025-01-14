import requests
from datetime import datetime
import boto3
import json
import time

# Configura o cliente do Kinesis Firehose
firehoseClient = boto3.client('firehose')  # Ajuste a região conforme necessário

def get_latest_crypto_price(coin):
    """
    Obtém o preço mais recente da criptomoeda usando a API CoinGecko.
    """
    url = f'https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=brl'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lança um erro se a requisição falhar
        data = response.json()
        return data[coin]['brl']
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar o preço via CoinGecko: {e}")
        return None

while True:
    try:
        # Captura o timestamp atual
        now = datetime.now()
        coleta = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Obtém o preço mais recente do Bitcoin
        price = get_latest_crypto_price('bitcoin')
        if price is not None:
            # Prepara os dados para envio
            data = {
                "price": price,
                "coleta": coleta
            }

            # Envia os dados para o Kinesis Firehose
            print(json.dumps({"price": price, "coleta": coleta}))
            envio = firehoseClient.put_record(
                DeliveryStreamName='btc_stream',  # Substitua pelo nome correto do seu Delivery Stream
                Record={
                    'Data': json.dumps(data) + "\n"  # Inclui uma quebra de linha para compatibilidade
                }
            )
            print(f"Enviado: {envio}, Preço: {price}, Coleta: {coleta}")
        else:
            print("Falha ao obter o preço. Tentando novamente...")

        # Aguarda 60 segundos antes da próxima coleta
        time.sleep(20)
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
        break
    except Exception as e:
        print(f"Erro inesperado: {e}")
        time.sleep(60)
