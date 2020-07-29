import statistics
from scipy import signal
from sklearn import cluster

IS_VERBOSE = False

class WaveformSlicer:

    def __init__(self):
        self.data = None
        self.filtered_data = None
        self.peaks = None
        self.wave_troughs = None
        self.wave_crests = None
        self.waves = None
        self.differences = None
        self.selected_wave_indexes = None
        self.status = None
        return


    def _find_filtered_data(self, data):
        indexes = [x for x, y in data]
        amplitudes = [y for x, y in data]
        b, a = signal.ellip(4, 0.01, 120, 0.125)
        filtered_amplitudes = signal.filtfilt(b, a, amplitudes, padlen = 50)
        filtered_data = [(index, filtered_amplitude) for index, filtered_amplitude in zip(indexes, filtered_amplitudes)]
        return filtered_data


    # 找出所有轉折點
    def _find_peaks(self, data):
        peaks = []
        previous_slope = None
        for index in range(1, len(data)):
            difference = data[index][1] - data[index - 1][1]
            if difference == 0:
                continue
            if difference > 0:
                if previous_slope is 'negative':
                    peaks.append(data[index - 1])
                previous_slope = 'positive'
                continue
            if difference < 0:
                if previous_slope is 'positive':
                    peaks.append(data[index - 1])
                previous_slope = 'negative'
                continue
        return peaks


    def _find_best_cluster_labels_and_centers(self, peaks):
        results = []
        for k in [2, 3]:
            kmeans = cluster.KMeans(n_clusters = k).fit([(0, peak[1]) for peak in peaks])
            cluster_labels = kmeans.labels_           # [label of peak, ...]
            cluster_centers = kmeans.cluster_centers_ # [[center of x in cluster 1, center of y in cluster 1], ...]

            centers = [cluster_centers[cluster_label][1] for cluster_label in cluster_labels]
            loss = 0
            for peak, center in zip(peaks, centers):
                loss += abs(peak[1] - center)
            results.append([cluster_labels, cluster_centers, loss])

            self._log(message = f'Loss (k = {k}): {loss}')

        if results[0][2] < 2 * results[1][2]: # loss(k = 2) < 2 x loss(k = 3)
            self._log(message = 'Selected k: 2')
            return (results[0][0], results[0][1])
        else:
            self._log(message = 'Selected k: 3')
            return (results[1][0], results[1][1])


    def _log(self, message):
        if IS_VERBOSE is True:
            print(message)
        return


    # 找出cluster_center最大的label(波峰的label)
    def _find_largest_cluster_label(self, cluster_centers):
        largest_cluster_label = 0
        largest_center = cluster_centers[0][1]
        for index in range(1, len(cluster_centers)):
            if cluster_centers[index][1] > largest_center:
                largest_cluster_label = index
                largest_center = cluster_centers[index][1]
        return largest_cluster_label


    # 找出cluster_center最小的label(波谷的label)
    def _find_smallest_cluster_label(self, cluster_centers):
        smallest_cluster_label = 0
        smallest_center = cluster_centers[0][1]
        for index in range(1, len(cluster_centers)):
            if cluster_centers[index][1] < smallest_center:
                smallest_cluster_label = index
                smallest_center = cluster_centers[index][1]
        return smallest_cluster_label


    # 找出波谷以及波峰的index(基於cluster_labels)
    def _find_wave_trough_crest_indexes(self, cluster_labels, wave_trough_label, wave_crest_label):
        wave_trough_indexes = []
        wave_crest_indexes = []
        for index in range(1, len(cluster_labels)):
            if cluster_labels[index - 1] == wave_trough_label and cluster_labels[index] == wave_crest_label:
                wave_trough_indexes.append(index - 1)
                wave_crest_indexes.append(index)
        return (wave_trough_indexes, wave_crest_indexes)


    def _find_filtered_peaks(self, peaks, filtration_indexes):
        filtered_peaks = []
        for index in range(len(peaks)):
            if index in filtration_indexes:
                filtered_peaks.append(peaks[index])
        return filtered_peaks


    # 根據切割波長的轉折點來切割data
    def _slice_data(self, data, slicing_peaks):
        waves = []
        slicing_peaks_x = [slicing_peak[0] for slicing_peak in slicing_peaks]
        for index in range(len(slicing_peaks_x) - 1):
            waves.append(data[slicing_peaks_x[index]:slicing_peaks_x[index + 1]])
        return waves


    def _find_selected_wave_indexes(self, differences):
        wave_lengths = differences['trough_to_trough_x']
        trough_to_crest_differences = differences['trough_to_crest_y']
        crest_to_trough_differences = differences['crest_to_trough_y']
        selected_wave_indexes = [wave_index for wave_index in range(1, len(self.waves) + 1)]
        for index in range(len(wave_lengths) - 1):
            if wave_lengths[index + 1] - wave_lengths[index] > wave_lengths[index] / 3:
                if index in selected_wave_indexes:
                    selected_wave_indexes.remove(index)
        for index in range(len(trough_to_crest_differences) - 1):
            if trough_to_crest_differences[index + 1] - trough_to_crest_differences[index] > trough_to_crest_differences[index] / 3:
                if index in selected_wave_indexes:
                    selected_wave_indexes.remove(index)
        for index in range(len(crest_to_trough_differences) - 1):
            if crest_to_trough_differences[index + 1] - crest_to_trough_differences[index] > crest_to_trough_differences[index] / 3:
                if index in selected_wave_indexes:
                    selected_wave_indexes.remove(index)
        return selected_wave_indexes


    def _determine_status(self, wave_lengths_std, selected_wave_indexes, waves):
        if wave_lengths_std >= 80:
            status = 0 # abnormal
        elif len(selected_wave_indexes) != len(waves):
            status = 1 # uncertain
        else:
            status = 2 # normal
        return status


    def fit(self, data):
        self.data = data                                                # data:          [(index, amplitude), ...]
        self.filtered_data = self._find_filtered_data(data = self.data) # filtered_data: [(index, amplitude), ...]
        self.peaks = self._find_peaks(data = self.filtered_data)        # peaks:         [(index, amplitude), ...]

        self.cluster_labels, self.cluster_centers = self._find_best_cluster_labels_and_centers(peaks = self.peaks)
        self.wave_crest_label = self._find_largest_cluster_label(cluster_centers = self.cluster_centers)
        self.wave_trough_label = self._find_smallest_cluster_label(cluster_centers = self.cluster_centers)
        self.wave_trough_indexes, self.wave_crest_indexes = self._find_wave_trough_crest_indexes(cluster_labels = self.cluster_labels, wave_trough_label = self.wave_trough_label, wave_crest_label = self.wave_crest_label)
        self.wave_troughs = self._find_filtered_peaks(peaks = self.peaks, filtration_indexes = self.wave_trough_indexes)
        self.wave_crests = self._find_filtered_peaks(peaks = self.peaks, filtration_indexes = self.wave_crest_indexes)
        self.waves = self._slice_data(data = self.data, slicing_peaks = self.wave_troughs)

        self.differences = dict()
        self.differences['trough_to_trough_x'] = [wave[-1][0] - wave[0][0] + 1 for wave in self.waves]
        self.differences['trough_to_crest_y'] = [self.wave_crests[index][1] - self.wave_troughs[index][1]     for index in range(len(self.wave_troughs) - 1)]
        self.differences['crest_to_trough_y'] = [self.wave_crests[index][1] - self.wave_troughs[index + 1][1] for index in range(len(self.wave_troughs) - 1)]

        self.selected_wave_indexes = self._find_selected_wave_indexes(self.differences)
        self.status = self._determine_status(statistics.stdev(self.differences['trough_to_trough_x']), self.selected_wave_indexes, self.waves)
        return


    def get_peaks(self):
        return self.peaks


    def get_wave_troughs(self):
        return self.wave_troughs


    def get_wave_crests(self):
        return self.wave_crests


    def get_waves(self):
        return self.waves


    def get_differences(self):
        return self.differences


    def get_status(self):
        return self.status


    def is_available(self):
        return True if self.status == 2 else False


    def get_selection(self):
        wave_boundaries = [[wave[0], wave[-1]] for wave in self.waves]
        if self.status == 2:
            selected_wave_boundaries = wave_boundaries
        elif self.status == 1:
            selected_wave_boundaries = [wave_boundaries[selected_wave_index - 1] for selected_wave_index in self.selected_wave_indexes]
        elif self.status == 0:
            selected_wave_boundaries = []
        return selected_wave_boundaries


if __name__ == '__main__':
    pass
