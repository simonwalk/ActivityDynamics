import sys
import os
import numpy as np
import datetime
import shutil
from util import prompt_with_timeout
from s01_extract_log_stackexchange import extract_log_stackexchange
from s02_basic_network_activity_analysis import basic_network_activity_analysis
from s02_basic_timeseries_activity_analysis import basic_timeseries_activity_analysis
from s03_generate_weighted_network import generate_weighted_network
from s04_threshold_filter_dataframe import threshold_filter_dataframe
from s04_time_gaps_analysis import time_gaps_analysis
from s05_core_activity_analysis import core_activity_analysis
from s05_extract_binned_posts_replies import extract_binned_posts_replies
from s06_calc_lambda import calc_lambda
from s06_calc_mu import calc_mu
from s07_calc_ratio import calc_ratio
from optparse import OptionParser


def now():
    return datetime.datetime.now()


def print_timestat(timestat):
    overall_time = timestat.values()[0]
    for i in timestat.values()[1:]:
        overall_time += i
    timestat = sorted(timestat.iteritems(), key=lambda x: x[1], reverse=True)
    print 'process time statistics'.center(100, '=')
    for name, time in timestat:
        print str(time), '\t' + str(name)
    print 'overall:', overall_time
    print '=' * 100


def run_all(log_filename, timestat=None, core=None, rolling_window_size=None, draw_network=None):
    folder = log_filename.rsplit('/', 1)[0] + '/'
    start = now()
    basic_network_activity_analysis(log_filename)
    if not timestat is None:
        timestat['basic network activity analysis'] = now() - start
    print 'log', log_filename
    print 'folder', folder
    optical_rolling_window_size = 60
    rolling_means = None
    fit_curve_deg = range(10)
    start = now()
    basic_timeseries_activity_analysis(log_filename, rolling_window=optical_rolling_window_size)
    if not timestat is None:
        timestat['basic time-series activity analysis'] = now() - start
    if draw_network is None:
        draw_network = prompt_with_timeout('Should Networks be visualized? (y/n)', timeout=5, default_val='n') == 'y'
    start = now()
    network_filename, threshold_results = generate_weighted_network(log_filename, draw=draw_network)
    if not timestat is None:
        timestat['generate weighted network'] = now() - start
    activity_values = sorted(threshold_results['activity'].values())
    activity_values_gaps = [activity_values[i + 1] - activity_values[i] for i, val in enumerate(activity_values[1:])]
    max_gap = (np.mean(activity_values_gaps) + np.median(activity_values_gaps)) / 2
    target_th = activity_values[0]
    for idx, i in enumerate(activity_values[1:]):
        if activity_values[idx + 1] - activity_values[idx] <= max_gap:
            target_th = i
    core_recommendation_th, preserved_activity = min(threshold_results['activity'].iteritems(), key=lambda x: abs(x[1] - target_th))
    if core is None:
        core = prompt_with_timeout('Enter Core-Size:', timeout=15, default_val=core_recommendation_th, target_data_type=int)
    start = now()
    threshold_filter_dataframe(log_filename)
    if not timestat is None:
        timestat['threshold filter dataframe'] = now() - start
    start = now()
    time_gaps_result = time_gaps_analysis(log_filename, core=core)
    recommendation = max(val['recommendation'] for val in time_gaps_result.values())
    recommendation_months = int(recommendation.total_seconds() / 2635200) + 1
    if not timestat is None:
        timestat['time gaps analysis'] = now() - start
    if rolling_window_size is None:
        rolling_window_size = prompt_with_timeout('Enter rolling window size in months:', timeout=15, default_val=recommendation_months, target_data_type=int)
    start = now()
    core_activity_analysis(log_filename, core=core)
    if not timestat is None:
        timestat['core activity analysis'] = now() - start
    start = now()
    extract_binned_posts_replies(log_filename, core=core)
    if not timestat is None:
        timestat['extract binned posts replies'] = now() - start
    start = now()
    calc_lambda(folder, rolling_window=rolling_window_size, rolling_means=rolling_means, fit_curve_deg=fit_curve_deg)
    if not timestat is None:
        timestat['calc lambda'] = now() - start
    start = now()
    calc_mu(folder, rolling_window=rolling_window_size, rolling_means=rolling_means, fit_curve_deg=fit_curve_deg)
    if not timestat is None:
        timestat['calc mu'] = now() - start
    start = now()
    calc_ratio(folder, net_filename=network_filename, rolling_means=rolling_means, fit_curve_deg=fit_curve_deg)
    if not timestat is None:
        timestat['calc ratio'] = now() - start
    print_timestat(timestat)


def run_all_stackexchange(folder, posts_file='Posts.xml', comments_file='Comments.xml', timestat=None, core=None, rolling_window_size=None, draw_network=None):
    start = now()
    log_filename = extract_log_stackexchange(folder, posts_file, comments_file)
    if not timestat is None:
        timestat['extract log file'] = now() - start
    run_all(log_filename, timestat=timestat, core=core, rolling_window_size=rolling_window_size, draw_network=draw_network)


def auto_decide(filename, core=None, rolling_window_size=None, draw_network=None):
    time_stat = dict()
    if filename.endswith('.7z') or os.path.isdir(filename):
        run_all_stackexchange(filename, timestat=time_stat, core=core, rolling_window_size=rolling_window_size, draw_network=draw_network)
    else:
        new_filename = filename + '_results/' + filename.rsplit('/', 1)[-1]
        try:
            os.mkdir(filename + '_results')
            os.system("cut -f 1-3 " + filename + " > " + new_filename)
        except Exception as e:
            print e.args
        filename = new_filename
        run_all(filename, timestat=time_stat, core=core, rolling_window_size=rolling_window_size, draw_network=draw_network)


if __name__ == '__main__':
    start = now()
    parser = OptionParser()
    parser.add_option("-c", action="store", type="int", dest="core")
    parser.add_option("-w", action="store", type="int", dest="rolling_window")
    parser.add_option("-d", action="store_true", dest="draw_network")
    (options, args) = parser.parse_args()
    core = 1
    rolling_window_size = 1
    draw_network = None
    auto_decide("/Users/simon/Desktop/Projects/DynamicNetworks/data_preparation/sorted_wikis/MathStackExchange/", core=core,
                rolling_window_size=1, draw_network=draw_network)
    print 'Overall Time:', str(now() - start)
    print 'ALL DONE -> EXIT'