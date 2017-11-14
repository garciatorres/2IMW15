import tweepy
import time
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tweepy import OAuthHandler
from collections import namedtuple

c_key = 'vux5IMAI9nilPH8dZgr9ZqyV0'
c_secret = 'W2dZt0dm6vSK17S3Q6zKOflvET5F9EtOSiY1QWtYpAK6sEvR4K'
a_token = '766751981230780416-h3zFfsN8Putuia4zUmbhjiMYCdNRB1X'
a_tokenSecret = 'Si8LXTlFmT1JjRAC02hHpAiYJWjuyxPgllHy2pwDpfc4M'
User = namedtuple('User', ['id', 'name', 'screen_name', 'verified', 'friends_count', 'statuses_count', 'favourites_count', 'followers_count', 'created_at', 'location', 'geo_enabled', 'protected', 'notifications', 'time_zone', 'description', 'has_extended_profile'])

auth = OAuthHandler(c_key, c_secret)
auth.set_access_token(a_token, a_tokenSecret)
api = tweepy.API(auth)

def trimIdList(idListToBetrimed, idListToBeCheck):
    for i in idListToBetrimed:
        if i not in idListToBeCheck:
            idListToBetrimed.remove(i)

def getUserInfo(id, w1, w2, w3):
    try:
        a = api.get_user(id)
        b = api.friends_ids(id)
        c = api.followers_ids(id)
        w1.writerow({'id': a.id, 'name': a.name, 'screen_name': a.screen_name, 'verified': a.verified, 'friends_count': a.friends_count, 'statuses_count': a.statuses_count, 'favourites_count': a.favourites_count, 'followers_count': a.followers_count, 'created_at': a.created_at, 'location': a.location, 'geo_enabled': a.geo_enabled, 'protected': a.protected, 'notifications': a.notifications, 'time_zone': a.time_zone, 'description': a.description, 'has_extended_profile': a.has_extended_profile})
        w2.writerow([id] + b)
        w3.writerow([id] + c)
    except tweepy.error.RateLimitError:
        print('Exceeding rate limit!! Wait 60 seconds...')
        time.sleep(60)
        return getUserInfo(id, w1, w2, w3)
    except tweepy.error.TweepError:
        return

    return

def main():
    fpr = open('./UserIDs.csv', newline='')
    reader = csv.reader(fpr)
    fpw1 = open('./userInfoData.csv', 'w')
    field_names = ['id', 'name', 'screen_name', 'verified', 'friends_count', 'statuses_count', 'favourites_count', 'followers_count', 'created_at', 'location', 'geo_enabled', 'protected', 'notifications', 'time_zone', 'description', 'has_extended_profile']
    writer1 = csv.DictWriter(fpw1, field_names, lineterminator='\n')
    writer1.writeheader()
    fpw2 = open('./userFollowingData.csv', 'w')
    writer2 = csv.writer(fpw2)
    fpw3 = open('./userFollowerData.csv', 'w')
    writer3 = csv.writer(fpw3)
    userList = []
    complete_cnt = 0
    for row in reader:
        getUserInfo(row[0], writer1, writer2, writer3)
        complete_cnt += 1
        print('{0} finished!'.format(complete_cnt))
    print("Done!")
    return

if __name__ == '__main__':
    main()
