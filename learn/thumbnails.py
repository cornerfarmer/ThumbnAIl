from model.Video import Video
import numpy as np

class ThumbnailDataset:

    def __init__(self):
        self.videos = Video.select()
        self.next_index = 0
        self.filenames, self.labels, self.max_views = self._videos_to_training_data(self.videos)

    def next_batch(self, batch_size):
        if self.next_index + batch_size < len(self.videos):
            batch = self.videos[self.next_index:self.next_index+batch_size]
            self.next_index += batch_size
        else:
            batch = self.videos[self.next_index:]
            self.next_index = 0
            batch += self.next_batch(batch_size - len(batch))

        return self._videos_to_training_data(batch)

    def _videos_to_training_data(self, videos):
        labels = np.zeros((len(videos), 1), dtype=np.float32)
        filenames = []
        for i, video in enumerate(videos):
            labels[i][0] = video.viewCount
            filenames.append('/home/domin/Dokumente/ThumbnAIl/thumbs/' + video.identifier + '.jpg')

        max_views = labels.max()
        labels /= labels.max()
        return [filenames, labels, max_views]

    def get_view_count_for_video(self, video_id):
        video = Video.select().where(Video.identifier == video_id).get()
        return video.viewCount