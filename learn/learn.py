import model.Config
model.Config.db_is_local = True

from model.BaseModel import db, api_key
from model.Video import Video

for video in Video.select():
    print(video.identifier)

print("Finished!")

db.close()