import random

import peewee

from model.Video import Video
import numpy as np
from math import log10

class ThumbnailDataset:

    def __init__(self, split_in_classes=False):
        self.split_in_classes = split_in_classes
        self.videos = Video.select()
        self.next_index = 0
        self._videos_to_training_data(self.videos)
        self._build_train_and_test_dataset()

    def next_batch(self, batch_size):
        if self.next_index + batch_size < len(self.videos):
            batch = self.videos[self.next_index:self.next_index+batch_size]
            self.next_index += batch_size
        else:
            batch = self.videos[self.next_index:]
            self.next_index = 0
            batch += self.next_batch(batch_size - len(batch))

        return self._videos_to_training_data(batch)

    def _build_train_and_test_dataset(self):
        test_start_index = int(len(self.filenames) * 0.8)
        self.train_filenames = self.filenames[:test_start_index]
        self.test_filenames = self.filenames[test_start_index:]
        self.train_labels = self.labels[:test_start_index]
        self.test_labels = self.labels[test_start_index:]


    def _videos_to_training_data(self, videos):
        labels = np.zeros((sum(len(video_set) for video_set in videos), 1 if not self.split_in_classes else len(videos)), dtype=np.float32)
        filenames = []

        video_index = 0
        for j, video_set in enumerate(videos):
            for video in video_set:
                if self.split_in_classes:
                    labels[video_index][j] = 1
                else:
                    labels[video_index][0] = self._get_label(video)
                filenames.append('/home/domin/Dokumente/ThumbnAIl/thumbs/' + video.identifier + '.jpg')
                video_index += 1

        max_views = labels.max()
        labels /= labels.max()
        self.filenames, self.labels, self.max_views = filenames, labels, max_views

    def get_view_count_for_video(self, video_id):
        video = Video.select().where(Video.identifier == video_id).get()
        return video.viewCount

    def _get_label(self, video):
        return video.viewCount

    def calculate_views_from_label(self, label):
        return self.max_views * label

class NormalizedThumbnailDataset(ThumbnailDataset):

    def __init__(self, split_in_classes=False, min_per_set=400):
        self.split_in_classes = split_in_classes
        max_views = Video.select(peewee.fn.Max(Video.viewCount)).scalar()
        self.videos = [[] for i in range(int(log10(max_views) + 1))]

        for video in Video.select().where(Video.viewCount > 0):
            self.videos[int(log10(video.viewCount)) if video.viewCount > 0 else 0].append(video)

        self.videos = [x for x in self.videos if not len(x) < min_per_set]
        smallest_set = min([len(x) for x in self.videos])

        self.videos = [random.sample(x, smallest_set) for x in self.videos]

        self.next_index = 0
        if self.split_in_classes:
            self._videos_to_training_data(self.videos)
        else:
            self._videos_to_training_data([[item for sublist in self.videos for item in sublist]])

        self._build_train_and_test_dataset()

    def _get_label(self, video):
        return log10(video.viewCount)

    def calculate_views_from_label(self, label):
        if self.split_in_classes:
            return pow(10, int(log10(self.videos[np.argmax(label)][0].viewCount)))
        else:
            return pow(10, self.max_views * label)



