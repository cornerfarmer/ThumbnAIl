from peewee import *

from model.BaseModel import BaseModel


class Channel(BaseModel):
    identifier = CharField(null=True, unique=True)
    name = CharField()
    viewedFeaturedChannels = BooleanField(default=False)
    subscriber = IntegerField(default=0)
    ownViewers = IntegerField(default=0)
