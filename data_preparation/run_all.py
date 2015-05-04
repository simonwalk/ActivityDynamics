__author__ = 'Florian Geigl, Simon Walk, Denis Helic'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

import os
import datetime
from s01_extract_log_stackexchange import extract_log_stackexchange
from s02_basic_network_activity_analysis import basic_network_activity_analysis
from s03_generate_weighted_network import generate_network
from s05_core_activity_analysis import core_activity_analysis
from s05_extract_binned_posts_replies import extract_binned_posts_replies
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


def run_all(log_filename, draw_network=None, mode="months"):
    print log_filename
    folder = log_filename.rsplit('/', 1)[0] + '/'
    basic_network_activity_analysis(log_filename)
    print 'log', log_filename
    print 'folder', folder
    generate_network(log_filename, draw=draw_network)
    core_activity_analysis(log_filename, core=0)
    extract_binned_posts_replies(log_filename, core=0, mode=mode)


def run_all_stackexchange(folder, posts_file='Posts.xml', comments_file='Comments.xml', timestat=None, draw_network=None, mode="months"):
    start = now()
    log_filename = extract_log_stackexchange(folder, posts_file, comments_file)
    if timestat is not None:
        timestat['extract log file'] = now() - start
    run_all(log_filename, draw_network=draw_network, mode=mode)


def auto_decide(filename, rolling_window_size=None, draw_network=None, mode="months"):
    time_stat = dict()
    if filename.endswith('.7z') or os.path.isdir(filename):
        run_all_stackexchange(filename, timestat=time_stat, draw_network=draw_network, mode=mode)
    else:
        new_filename = filename + '_results/' + filename.rsplit('/', 1)[-1]
        try:
            os.mkdir(filename + '_results')
            os.system("cut -f 1-3 " + filename + " > " + new_filename)
        except Exception as e:
            print e.args
        filename = new_filename
        run_all(filename, draw_network=draw_network, mode=mode)


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
    root_path = "../datasets/"
    #path = root_path + "Beachapedia/beachapedia_org_collab_network.txt.sorted"
    #path = root_path + "Nobbz/nobbz_org_collab_network.txt.sorted"
    #path = root_path + "W15M/wiki_15m_cc_collab_network.txt.sorted"
    #path = root_path + "CharacterDB/characterdb_cjklib_org_collab_network.txt.sorted"
    path = root_path + "EnglishStackExchange/"
    auto_decide(path, rolling_window_size=1, draw_network=draw_network, mode="months")
    print 'Overall Time:', str(now() - start)
    print 'ALL DONE -> EXIT'