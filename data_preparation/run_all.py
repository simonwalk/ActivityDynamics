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
    basic_network_activity_analysis(log_filename)
    print 'log', log_filename
    print 'folder', folder
    generate_weighted_network(log_filename, draw=draw_network)
    core_activity_analysis(log_filename, core=0)
    extract_binned_posts_replies(log_filename, core=0)


def run_all_stackexchange(folder, posts_file='Posts.xml', comments_file='Comments.xml', timestat=None, core=None, rolling_window_size=None, draw_network=None):
    start = now()
    log_filename = extract_log_stackexchange(folder, posts_file, comments_file)
    if timestat is not None:
        timestat['extract log file'] = now() - start
    run_all(log_filename, timestat=timestat, core=core, rolling_window_size=rolling_window_size, draw_network=draw_network)


def auto_decide(filename, core=0, rolling_window_size=None, draw_network=None):
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
    core = 0
    rolling_window_size = 1
    draw_network = None
    auto_decide("/opt/datasets/stackexchange/english.stackexchange.com.7z", core=core,
                rolling_window_size=1, draw_network=draw_network)
    #auto_decide("/Volumes/DataStorage/Programming/BeerStackExchange/", core=core,
    #            rolling_window_size=1, draw_network=draw_network)
    print 'Overall Time:', str(now() - start)
    print 'ALL DONE -> EXIT'