import sqlite3
import pandas as pd

def get_typhoon_data():
    conn = sqlite3.connect('data/social_media/yolandatweets.sqlite')

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tweets")

    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=['tweet', 'date'])
    df.astype({'tweet': 'str', 'date': 'str'})
    df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %z %Y")

    cursor.close()
    conn.close()

    return df




