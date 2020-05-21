import csv
import os
import matplotlib.pyplot as plot
import statistics
from sklearn.metrics import mean_squared_error
from waveform_slicer import WaveformSlicer

ROOT_PATHS = ['datasets/1000/', 'datasets/1015/']

def read_directories(root_path):
    for root, directories, file_names in os.walk(root_path):
        return [root + directory + '/' for directory in directories]


def read_samples(directories):
    samples = []
    for directory in directories:
        sample = []
        for root, dirs, file_names in os.walk(directory):
            file_names.sort()
            for file_name in file_names:
                if file_name[-4:] == '.FFT':
                    file = open(directory + file_name, 'r', encoding = 'utf-8')
                    lines = file.readlines()
                    for index in range(1, len(lines)):
                        sample.append(int(lines[index].replace('\n', '')))
        sample = [(index, sample[index]) for index in range(len(sample))] # add index for each point
        samples.append(sample)
    return samples


def plot_time_domain_amplitude(data, peaks, slicing_peaks):
    plot.xlabel('Time')
    plot.ylabel('Amplitude')
    plot.plot([x for x, y in data], [y for x, y in data], color = 'cornflowerblue')      # plot original data
    plot.scatter([x for x, y in peaks], [y for x, y in peaks], color = 'mediumseagreen') # plot peaks
    for slicing_peak in slicing_peaks:
        plot.axvline(x = slicing_peak[0], color = 'grey', linestyle = '--')              # plot slicing lines
    plot.show()
    return


def write_file(file, dataset_name, wave_lengths_mean, wave_lengths_std, standard_wave_length, loss, loss_2):
    file.write('{}, {}, {}, {}, {}, {}\n'.format(dataset_name, wave_lengths_mean, wave_lengths_std, standard_wave_length, loss, loss_2))
    return


if __name__ == '__main__':

    directories = []
    for root_path in ROOT_PATHS:
        directories += read_directories(root_path)
    samples = read_samples(directories)

    output_file = open('result.csv', 'w', encoding = 'utf_8_sig')

    for sample_index, sample in enumerate(samples):
        dataset_name = directories[sample_index]
        print(dataset_name)

        if len(sample) == 0:
            print('skipped')
            continue

        sample = sample[1500:]

        # heartbeat
        file = open(dataset_name + '001full.csv', 'r', encoding = 'utf-8')
        rows = list(csv.reader(file))
        file.close()
        try:
            heartbeat = int(rows[0][3].split('/')[1].split(' ')[1])
        except:
            print('skipped')
            continue
        standard_wave_length = heartbeat / 60 * 500

        # data: [(index, amplitude), ...]
        slicer = WaveformSlicer()
        slicer.fit(sample)
        peaks = slicer.get_peaks()
        slicing_peaks = slicer.get_slicing_peaks()

        # plot_time_domain_amplitude(data, peaks, slicing_peaks)

        wave_lengths = []
        for index in range(len(slicing_peaks) - 1):
            wave_lengths.append(slicing_peaks[index + 1][0] - slicing_peaks[index][0])

        wave_lengths_mean = statistics.mean(wave_lengths)
        wave_lengths_std = statistics.stdev(wave_lengths)
        standard_wave_lengths = [standard_wave_length] * len(wave_lengths)
        loss = mean_squared_error(standard_wave_lengths, wave_lengths)
        loss_2 = (standard_wave_length - sum(wave_lengths) / len(wave_lengths)) / len(wave_lengths)

        print(wave_lengths)
        print('Mean:', wave_lengths_mean)
        print('Std:', wave_lengths_std)
        print('Heartbeat length:', standard_wave_length)
        print('Loss (MSE):', loss)
        print('Loss (ME):', loss_2)

        write_file(output_file, dataset_name.split('/')[2], wave_lengths_mean, wave_lengths_std, standard_wave_length, loss, loss_2)
    output_file.close()
