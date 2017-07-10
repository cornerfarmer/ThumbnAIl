import sys

sys.path.append("/kunden/homepages/42/d584324863/htdocs/youtubemap/crawler/modules/")

from model.BaseModel import db
from model.Video import Video
from urllib.request import urlretrievey

try:

    db.connect()
    db.create_tables([Video], True)

    for video in Video.select():

        url = "https://i.ytimg.com/vi/" + video.identifier + "/default.jpg"
        urlretrieve(url, "../thumbs/" + video.identifier + ".jpg")

        print("Added thumbnail for " + video.identifier)

except:
    db.close()
    raise

print("Finished!")

db.close()