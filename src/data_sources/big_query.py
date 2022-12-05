from google.cloud import bigquery
import os

DATASET=os.environ.get('DATASET')
PROJECT=os.environ.get('PROJECT')
TABLE_01=os.environ.get('TABLE_01')
CHUNK_SIZE = os.environ.get('CHUNK_SIZE')



def get_chunk_from_bq():
    '''
    Fonction pour que le package puisse
    accèder à la data tabulaire sur Big Query.
    '''

    table = f"{PROJECT}.{DATASET}.{TABLE_01}"
    client = bigquery.Client()
    x = 0
    for x in range(0, TABLE_01.size - CHUNK_SIZE):
        rows = client.list_rows(table, start_index= x, max_results= x + CHUNK_SIZE)
        big_query_df = rows.to_dataframe()
        x = x + CHUNK_SIZE

    if big_query_df.shape[0] == 0:
        return None  # end of data

    return big_query_df


def save_bq_chunk():

    """
    save a chunk of the raw dataset to big query
    empty the table beforehands if `is_first` is True

    """
