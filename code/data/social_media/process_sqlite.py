import json
import math
import sqlite3
from datetime import datetime
import datetime as dt
import pandas as pd
import pytz


def get_typhoon_data():
    conn = sqlite3.connect('data/social_media/yolandatweets.sqlite')

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tweets")

    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=['tweet', 'date'])
    df.astype({'tweet': 'str', 'date': 'str'})
    df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %z %Y")
    df["date"] = df["date"].apply(lambda x: x.replace(tzinfo=pytz.UTC))
    cursor.close()
    conn.close()

    return df


def get_florence():
    frames = []
    for i in range(1, 31):
        records = map(json.loads, open(f'data/social_media/florence/2018-09-{str(i).zfill(2)}.ndjson', encoding="utf8"))
        df = pd.DataFrame.from_records(records)
        frames.append(df)

    result = pd.concat(frames)
    result["date"] = pd.to_datetime(result["created_at"], format="%Y-%m-%dT%H:%M:%S")
    result["date"] = result["date"].apply(lambda x: x.replace(tzinfo=pytz.UTC))
    result["tweet"] = result["text"]
    result = result.drop(columns=['id', 'text', 'conversation_id', 'lang', 'geo', 'user', 'media', 'created_at'])
    return result


def get_geoloc():
    frames = []
    for i in range(6, 7):
        records = map(json.loads, open(f'data/social_media/geoloc/2020-12-{str(i).zfill(2)}.ndjson', encoding="utf8"))
        df = pd.DataFrame.from_records(records)
        frames.append(df)
    if len(frames) > 1:
        result = pd.concat(frames)
    else:
        result = frames[0]
    result["date"] = pd.to_datetime(result["created_at"], format="%Y-%m-%dT%H:%M:%S")
    result["date"] = result["date"].apply(lambda x: x.replace(tzinfo=pytz.UTC))
    result["tweet"] = result["text"]
    result = result.drop(columns=['id', 'text', 'conversation_id', 'lang', 'geo', 'user', 'media', 'created_at'])
    return result


def get_date(time):
    frac, whole = math.modf(time)
    mins = 60 * frac
    return datetime(2016, 9, 12, int(whole), round(mins), 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)


def get_weibo():
    input = pd.read_csv("data/social_media/other/weibo.csv")
    df = pd.DataFrame()
    df["date"] = input["release_time"].apply(lambda x: get_date(x))
    df["tweet"] = input["weibo_id"]
    return df


def get_txt_tweet(hashtag, start_date):
    hashtags = []
    with open('other/twt_ts.txt', 'r', encoding="utf8") as f:
        avalanches = f.readlines()
    with open('other/twt_hashtags.txt', 'r', encoding="utf8") as f:
        for h in f.readlines():
            hashtags.append(h.replace("\n", ""))

    index = hashtags.index(hashtag)
    values = avalanches[index].split(" ")[1:-1]
    df = pd.DataFrame()
    df["values"] = values
    df = df.assign(date=pd.date_range(start=start_date, periods=len(values), tz='utc', freq="1s"))
    return df


if __name__ == "__main__":
    hashtag = "Women4Bernie"
    start_date = datetime(2013, 11, 7, 17, 0, 0, tzinfo=dt.timezone.utc) \
        .replace(tzinfo=pytz.UTC)
    get_txt_tweet(hashtag, start_date, freq="15min")
