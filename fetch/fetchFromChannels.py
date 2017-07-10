import sys

sys.path.append("/kunden/homepages/42/d584324863/htdocs/youtubemap/crawler/modules/")

from apiclient.discovery import build
from model.BaseModel import db, api_key
from model.Channel import Channel
from model.Video import Video

try:

    db.connect()
    db.create_tables([Channel, Video], True)

    service = build('youtube', 'v3', developerKey=api_key)

    for channel in Channel.select():

        print("Checking for new videos on channel " + channel.name)

        request = service.search().list(part='snippet', maxResults='50', channelId=channel.identifier, order="date", type="video", publishedAfter="2017-01-01T00:00:00Z", fields='items/id/videoId')
        results = request.execute()

        addedVideos = 0
        for item in results["items"]:
            videoId = item["id"]["videoId"]

            if Video.select().where(Video.identifier == videoId).count() == 0:
                video = Video(identifier=videoId, channel=channel)
                video.save(force_insert=True)
                addedVideos += 1

        print("Found " + str(addedVideos) + " new Videos")

except:
    db.close()
    raise

print("Finished!")

db.close()