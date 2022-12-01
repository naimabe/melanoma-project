from google.cloud import bigquery
import os

DATASET=os.environ.get('DATASET')
PROJECT=os.environ.get('PROJECT')
TABLE_01=os.environ.get('TABLE_01')
CHUNK_SIZE = os.environ.get('CHUNK_SIZE')


def get_data_from_bq():
    '''
    Fonction pour que le package puisse
    accèder à la data tabulaire sur Big Query.

    '''
    table = f"{PROJECT}.{DATASET}.{TABLE_01}"

    client = bigquery.Client()
    rows = client.list_rows(table, start_index=CHUNK_SIZE, max_results= x + CHUNK_SIZE)
    df = rows.to_dataframe()
    ''' dsakjhfkjadh'''
