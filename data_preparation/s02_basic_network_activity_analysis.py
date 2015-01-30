from __future__ import division
from util import *
from collections import defaultdict


def basic_network_activity_analysis(log_filename):
    print_f('start basic network activity analysis')
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    print_f('process: ', log_filename)
    communication_dict = defaultdict(lambda: dict())
    histo_dict = defaultdict(int)
    activity = 0
    links = 0
    with open(log_filename, 'r') as f:
        for counter, line in enumerate(f):
            if counter % 1000000 == 0:
                print_f("processed", counter / 1000000, "m")
            activity += 1
            line = line.strip().split('\t')
            if len(line) > 2:
                src, dest = int(line[1]), int(line[2])
                src, dest = (src, dest) if src < dest else (dest, src)
                try:
                    communication_dict[src][dest] += 1
                except KeyError:
                    links += 1
                    communication_dict[src][dest] = 1

    print_f('create histogram data')
    for src_item, dest_dict, in communication_dict.iteritems():
        for dest_item, val in dest_dict.iteritems():
            histo_dict[val] += 1

    # print_f('plot')
    # max_key = max(histo_dict.keys())
    # x = np.arange(max_key + 1)
    # y = [histo_dict[i] for i in x]
    # plt.clf()
    # plt.plot(x, y)
    # plt.grid(axis='y')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('activity per link')
    # plt.ylabel('#links')
    # plt.savefig('communication_th.png', dpi=150)
    # plt.close('all')

    print 'all links:', links
    print 'all activity:', activity
    links_th = dict()
    activity_th = dict()
    for i in range(0, 10):
        c_links = links - sum([histo_dict[j] for j in range(0, i + 1)])
        c_activity = activity - sum([histo_dict[j] * j for j in range(0, i + 1)])
        links_th[i] = c_links
        activity_th[i] = c_activity

    log = False
    for norm_flag in [False, True]:
        if norm_flag:
            norm = 1 / links * 100
            links_th = {key: val * norm for key, val in links_th.iteritems()}

            norm = 1 / activity * 100
            activity_th = {key: val * norm for key, val in activity_th.iteritems()}

        # plt.clf()
        # plt.figure()
        # fig, ax = plt.subplots()
        # print_f('plot threshold')
        # ax.plot(*zip(*sorted(links_th.iteritems(), key=lambda x: x[0])), label='links', c='green', linewidth=2)
        # if log:
        #     ax.set_yscale('log')
        # if norm_flag:
        #     ax.set_ylabel('links/activity')
        #     ax2 = ax
        # else:
        #     ax.set_ylabel('links')
        #     ax2 = ax.twinx()
        # ax2.plot(*zip(*sorted(activity_th.iteritems(), key=lambda x: x[0])), label='activity', c='blue', linewidth=2)
        # if not norm_flag:
        #     ax2.set_ylabel('activity')
        # if log:
        #     ax2.set_yscale('log')
        # plt.xlabel('threshold')
        # ax.legend(loc=2)
        # ax2.legend(loc=1)
        # if norm_flag:
        #     ax.set_ylim([0, 100])
        #     ax2.set_ylim([0, 100])
        # # plt.legend()
        # ax.grid()
        # ax.set_xlabel('activity threshold')
        # plot_name = 'activity_threshold_analysis' if not norm_flag else 'activity_threshold_analysis_norm'
        # plot_name = plot_name + '.png'
        # plt.savefig(folder_name + plot_name, dpi=150)
        # plt.close('all')
    print_f('basic network activity analysis done')


if __name__ == '__main__':
    start = datetime.datetime.now()
    basic_network_activity_analysis("/opt/datasets/fitnessexchange/extracted_logfile" if len(sys.argv) <= 1 else sys.argv[1])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'