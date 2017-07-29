import sys

sys.path.append("/kunden/homepages/42/d584324863/htdocs/youtubemap/crawler/modules/")
sys.path.append("/kunden/homepages/42/d584324863/htdocs/thumbnAIls")

from apiclient.discovery import build
from model.BaseModel import db, api_key
from model.Channel import Channel
from model.Video import Video

try:

    db.connect()
    db.create_tables([Channel, Video], True)

    service = build('youtube', 'v3', developerKey=api_key)

    for video in Video.select().where(Video.viewCount==0):

        request = service.videos().list(part='statistics', id=video.identifier)
        results = request.execute()

        for item in results["items"]:
            video.viewCount = item["statistics"]["viewCount"]
            video.save()

            print(video.identifier + " -> " + video.viewCount)

except:
    db.close()
    raise

print("Finished!")

db.close()