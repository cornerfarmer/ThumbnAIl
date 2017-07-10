from model.BaseModel import BaseModel
from peewee import *

from model.Channel import Channel


class Video(BaseModel):
    identifier = CharField()
    channel = ForeignKeyField(Channel)
    viewCount = IntegerField(default=0)

