import json
import tweepy


consumer_key = "AAAA"
consumer_secret = "AAAA"
access_token  = "AAAA"
access_token_secret = "AAAA"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def lookup_tweets(file_in_ids,file_out_json):
    # tweet ids to look up
    list_tweet_ids = [line.rstrip('\n') for line in open(file_in_ids)]
    with open(file_out_json,'w') as f_out:
        Number_tweets_retrieved = 0
        tweet_count = len(list_tweet_ids)
        try:
            for i in range((tweet_count // 100)+ 1):#
                end_loc = min((i + 1) * 100, tweet_count)
                temp =  api.statuses_lookup(id_=list_tweet_ids[i * 100:end_loc])
                Number_tweets_retrieved += len(temp)
                for tweet in temp:
                    f_out.write(json.dumps(tweet._json))
                    f_out.write('\n')
                print  ("Number of tweets collected: {0} ".format(Number_tweets_retrieved))
        except tweepy.TweepError:
            print ('Something went wrong, quitting...')

if __name__ == '__main__':
    lookup_tweets('Firstweek_all_id.txt','Firstweek_all_tweets_in_json.json')