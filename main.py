import csv
import os
import math
import matplotlib.pyplot as plot
import statistics
import time
from sklearn.metrics import mean_squared_error
from waveform_slicer import WaveformSlicer

STATISTICS_FILE_HEADER = ['folder_name', 'L/R', 'age', 'gender', 'number_of_waves',
                          'valleys_distance_avg', 'valleys_distance_std', 'valleys_distance_from_heartbeat', 'valleys_distance_from_pulse',
                          'difference_between_heartbeat_and_average(mse)', 'difference_between_heartbeat_and_average(mse_sqrt)', 'difference_between_heartbeat_and_average(me)',
                          'difference_between_pulse_and_average(mse)', 'difference_between_pulse_and_average(mse_sqrt)', 'difference_between_pulse_and_average(me)']
RESULT_FILE_HEADER = ['folder_name', 'number_of_waves', 'valleys_x']
FILE_HEADER = ['folder_name', 'L/R', 'age', 'gender', 'number_of_waves', 'crests_x', 'valleys_x',
               'wavelengths_avg', 'wavelengths_std', 'wavelengths(max-min)',
               'amplitudes_v2c_avg', 'amplitudes_v2c_std', 'amplitudes_v2c(max-min)',
               'amplitudes_c2v_avg', 'amplitudes_c2v_std', 'amplitudes_c2v(max-min)']
CLASSIFICATION_FILE_HEADER = ['class', 'folder_names']

# ROOT_PATHS = ['datasets/1000/', 'datasets/1015/', 'datasets/Archive/']
ROOT_PATHS = ['datasets/Archive/']

START_TIME = int(time.time())

STOPPING_LIST = []

BIAS = 1500

IS_FIGURE_SHOWN = False

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
                    file.close()
                    for index in range(1, len(lines)):
                        sample.append(int(lines[index].replace('\n', '')))
        sample = [(index, sample[index]) for index in range(len(sample))] # add index for each point
        samples.append(sample)
    return samples


def read_information(dataset_path):
    information = []
    try:
        rows = read_csv_file(dataset_path + '001full.csv')
        id = rows[0][0]
        gender = rows[0][1]
        age = rows[0][2]
        heartbeat = int(rows[0][3].split('/')[1].split(' ')[1])
        if '-R' in id:
            information.append('R')
        else:
            information.append('L')
        information.append(age)
        information.append(gender)
        information.append(heartbeat)
        pulse_frequencies = []
        for index in range(4):
            rows = read_csv_file(dataset_path + f'00{str(index + 1)}full.csv')
            pulse_frequencies.append(float(rows[1][10]))
        information.append(statistics.mean(pulse_frequencies))
    except:
        information = None
    return information


def plot_time_domain_amplitude(data, peaks, slicing_peaks, patient_information, is_figure_shown):
    patient_information_text = 'folder_ID: {}\nposition: {}\ngender: {}\nage: {}\nheartbeat: {}, pulse: {}'.format(*patient_information)
    plot.xlabel('Time')
    plot.ylabel('Amplitude')
    plot.plot([x for x, y in data], [y for x, y in data], color = 'cornflowerblue')      # plot original data
    plot.scatter([x for x, y in peaks], [y for x, y in peaks], color = 'mediumseagreen') # plot peaks
    for slicing_peak in slicing_peaks:
        plot.axvline(x = slicing_peak[0], color = 'grey', linestyle = '--')              # plot slicing lines
    plot.text(0.8, 0.9, patient_information_text, fontsize = 9, transform = plot.gcf().transFigure)
    if is_figure_shown is True:
        plot.show()
    else:
        plot.gcf().set_size_inches(16, 9)
        plot.savefig(f'results_{START_TIME}/{patient_information[0]}.jpg')
        plot.close()
    return


def read_csv_file(file_path):
    file = open(file_path, 'r', encoding = 'utf-8')
    rows = list(csv.reader(file))
    file.close()
    return rows


def write_csv_file(file, values):
    file.write(','.join([str(value) for value in values]) + '\n')
    return


if __name__ == '__main__':

    directories = []
    for root_path in ROOT_PATHS:
        directories += read_directories(root_path)
    samples = read_samples(directories)

    os.mkdir(f'results_{START_TIME}')
    statistics_file = open(f'results_{START_TIME}/result.csv', 'w', encoding = 'utf_8_sig')
    result_file = open(f'results_{START_TIME}/result2.csv', 'w', encoding = 'utf_8_sig')
    file = open(f'results_{START_TIME}/result3.csv', 'w', encoding = 'utf_8_sig')
    classification_file = open(f'results_{START_TIME}/result4.csv', 'w', encoding = 'utf_8_sig')

    write_csv_file(statistics_file, STATISTICS_FILE_HEADER)
    write_csv_file(result_file, RESULT_FILE_HEADER)
    write_csv_file(file, FILE_HEADER)
    write_csv_file(classification_file, CLASSIFICATION_FILE_HEADER)

    categories = dict()
    categories['abnormal'] = []
    categories['uncertain'] = []
    categories['normal'] = []

    for sample_index, sample in enumerate(samples):
        dataset_path = directories[sample_index]
        dataset_name = dataset_path.split('/')[2]
        print(dataset_path)

        if len(STOPPING_LIST) > 0 and dataset_name not in STOPPING_LIST:
            continue

        if len(sample) == 0:
            print('skipped')
            continue

        sample = sample[BIAS:]
        sample = [(index, sample[index][1]) for index in range(len(sample))]

        # information
        information = read_information(dataset_path)
        if information is None:
            print('skipped')
            continue
        hand, age, gender, heartbeat, pulse_frequency = information
        standard_wave_length = 60 / heartbeat * 500
        standard_pulse_wave_length = 1 / pulse_frequency * 500

        # data: [(index, amplitude), ...]
        slicer = WaveformSlicer()
        slicer.fit(sample)
        peaks = slicer.get_peaks()
        slicing_peaks = slicer.get_wave_troughs()
        wave_crests = slicer.get_wave_crests()

        patient_information = dataset_name, hand, gender, age, heartbeat, round(pulse_frequency * 60)
        plot_time_domain_amplitude(sample, peaks, slicing_peaks, patient_information, IS_FIGURE_SHOWN)

        differences = slicer.get_differences()
        wave_lengths = differences['trough_to_trough_x']
        t2c_differences_y = differences['trough_to_crest_y']
        c2t_differences_y = differences['crest_to_trough_y']
        t2c_differences_y_mean = statistics.mean(t2c_differences_y)
        t2c_differences_y_std = statistics.stdev(t2c_differences_y)
        c2t_differences_y_mean = statistics.mean(c2t_differences_y)
        c2t_differences_y_std = statistics.stdev(c2t_differences_y)

        number_of_waves = len(wave_lengths)
        wave_lengths_mean = statistics.mean(wave_lengths)
        wave_lengths_std = statistics.stdev(wave_lengths)
        standard_wave_lengths = [standard_wave_length] * number_of_waves
        standard_pulse_wave_lengths = [standard_pulse_wave_length] * number_of_waves
        loss_mse = mean_squared_error(standard_wave_lengths, wave_lengths)
        loss_mse_pulse = mean_squared_error(standard_pulse_wave_lengths, wave_lengths)
        loss_mse_sqrt = math.sqrt(loss_mse)
        loss_mse_sqrt_pulse = math.sqrt(loss_mse_pulse)
        loss_me = (standard_wave_length - sum(wave_lengths) / number_of_waves) / number_of_waves
        loss_me_pulse = (standard_pulse_wave_length - sum(wave_lengths) / number_of_waves) / number_of_waves

        slicing_peak_indexes = [str(slicing_peak[0] - BIAS) for slicing_peak in slicing_peaks]
        slicing_peak_indexes = ', '.join(slicing_peak_indexes)
        wave_crest_indexes = [str(wave_crest[0] - BIAS) for wave_crest in wave_crests]
        wave_crest_indexes = ', '.join(wave_crest_indexes)

        if slicer.get_status() == 0:
            categories['abnormal'].append(dataset_name)
        elif slicer.get_status() == 1:
            categories['uncertain'].append(dataset_name)
        elif slicer.get_status() == 2:
            categories['normal'].append(dataset_name)

        write_csv_file(statistics_file, [dataset_name, hand, age, gender, number_of_waves,
                                         round(wave_lengths_mean), round(wave_lengths_std), round(standard_wave_length), round(standard_pulse_wave_length),
                                         round(loss_mse), round(loss_mse_sqrt), round(loss_me),
                                         round(loss_mse_pulse), round(loss_mse_sqrt_pulse), round(loss_me_pulse)])
        write_csv_file(result_file, [dataset_name, len(slicing_peaks) - 1, f'"{slicing_peak_indexes}"'])
        write_csv_file(file, [dataset_name, hand, age, gender, number_of_waves, f'"{wave_crest_indexes}"', f'"{slicing_peak_indexes}"',
                              round(wave_lengths_mean), round(wave_lengths_std), max(wave_lengths) - min(wave_lengths),
                              round(t2c_differences_y_mean), round(t2c_differences_y_std), round(max(t2c_differences_y) - min(t2c_differences_y)),
                              round(c2t_differences_y_mean), round(c2t_differences_y_std), round(max(c2t_differences_y) - min(c2t_differences_y))])
    statistics_file.close()
    result_file.close()
    file.close()

    for category, dataset_names in categories.items():
        dataset_names = ', '.join(dataset_names)
        write_csv_file(classification_file, [category, f'"{dataset_names}"'])
    classification_file.close()
