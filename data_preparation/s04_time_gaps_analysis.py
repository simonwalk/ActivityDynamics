from __future__ import division
from util import *


def bin_date(timestamp):
    # return timestamp.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    return timestamp.replace(microsecond=0, second=0, minute=0, hour=0)


def get_stat_timegaps(data):
    data = [x.to_pytimedelta().total_seconds() for x in (filter(lambda x: not isinstance(x, pd.tslib.NaTType), data))]
    min_ = np.min(data)
    max_ = np.max(data)
    mean = np.mean(data)
    median = np.median(data)
    stddev = np.std(data)
    min_, max_, mean, median, stddev = [datetime.timedelta(seconds=x) for x in [min_, max_, mean, median, stddev]]
    result = dict()
    result['min'] = min_
    result['max'] = max_
    result['mean'] = mean
    result['median'] = median
    result['stddev'] = stddev
    result['recommendation'] = (mean + stddev + stddev)
    return result


def get_stat_bursts(list_of_data):
    mean = np.nanmean(list_of_data)
    median = np.nanmedian(list_of_data)
    stddev = np.nanstd(list_of_data)
    min_ = np.nanmin(list_of_data)
    max_ = np.nanmax(list_of_data)
    result = dict()
    result['min'] = min_
    result['max'] = max_
    result['mean'] = mean
    result['median'] = median
    result['stddev'] = stddev
    return result


def time_gaps_analysis(log_filename, net_filename=None, core=1, sample_size=-1):
    print_f('start time gaps analysis')
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    network_name = folder_name + 'weighted_net.gt' if net_filename is None else net_filename

    data_frame = read_log_to_df(log_filename)
    df_start_time, df_end_time = data_frame['timestamp'].min(), data_frame['timestamp'].max()
    df_duration = (df_end_time - df_start_time).total_seconds()
    print_f('bin source')
    src_bin_df = data_frame.groupby('source')
    print_f('done')

    if core > 0:
        graph = read_graph(network_name, activity_threshold=core, largest_component=True)

    print_f('create set of user-ids')
    if core > 0:
        user_ids = {int(graph.vertex_properties["nodeID"][v]) for v in graph.vertices()}
        print_f('core', core, 'contains:', len(user_ids), 'users')
    else:
        user_ids = {int(i) for i in data_frame['source']}
        print_f('dataset contains', len(user_ids), 'users')

    if sample_size == -1 or sample_size >= len(user_ids):
        user_ids = set(user_ids)
    else:
        print_f('sample', sample_size, 'users')
        user_ids = set(random.sample(user_ids, sample_size))

    percent = -1
    len_users = len(user_ids)
    process_times = []
    dropped_users = 0
    prog = 0
    time_gaps_posts = []
    time_gaps_replies = []
    time_gaps_all = []
    start = datetime.datetime.now()
    bursts_all = []
    bursts_posts = []
    bursts_replies = []
    for user, gdf in src_bin_df:
        user = int(user)
        if user in user_ids:
            prog += 1
            # all time diffs
            user_timestamps = gdf['timestamp']
            user_activity_span = user_timestamps.max() - user_timestamps.min()
            # print 'uas:', user_activity_span, type(user_activity_span)
            if isinstance(user_activity_span, datetime.timedelta):
                bursts_all.append(user_activity_span.total_seconds() / df_duration * 100)
            user_time_gaps = user_timestamps.diff()
            time_gaps_all.extend(user_time_gaps)
            # post time diffs
            posts_idx = np.isnan(gdf['destination'])
            user_posts_timestamps = gdf['timestamp'][posts_idx]
            user_posts_span = user_posts_timestamps.max() - user_posts_timestamps.min()
            # print 'ups:', user_posts_span, type(user_posts_span)
            if isinstance(user_posts_span, datetime.timedelta):
                bursts_posts.append(user_posts_span.total_seconds() / df_duration * 100)
            user_time_gaps_posts = user_posts_timestamps.diff()
            time_gaps_posts.extend(user_time_gaps_posts)
            # replies time diffs
            replies_idx = np.invert(posts_idx)
            user_replies_timestamps = gdf['timestamp'][replies_idx]
            user_replies_span = user_replies_timestamps.max() - user_replies_timestamps.min()
            # print 'urs:', user_replies_span, type(user_replies_span)
            if isinstance(user_replies_span, datetime.timedelta):
                bursts_replies.append(user_replies_span.total_seconds() / df_duration * 100)
            user_time_gaps_replies = user_replies_timestamps.diff()
            time_gaps_replies.extend(user_time_gaps_replies)

            process_times.append((datetime.datetime.now() - start).total_seconds())
            start = datetime.datetime.now()
        # status update
        tmp_percent = int(prog / len_users * 100)
        process_times = process_times[-1000:]
        if tmp_percent > percent:
            percent = tmp_percent
            print_f('processing...', str(percent).rjust(3, '0'), '%', 'users:', prog)
            if process_times:
                avg = np.mean(process_times)
                print '\tavg time per user in sec:', datetime.timedelta(seconds=avg).total_seconds(), 'est remain:', datetime.timedelta(seconds=avg * (len_users - prog))
                if dropped_users > 0:
                    print '\tdropped users:', int(dropped_users / prog * 100), '%'

    bursts_analysis = dict()
    bursts_analysis['all activity'] = get_stat_bursts(bursts_all)
    bursts_analysis['posts'] = get_stat_bursts(bursts_posts)
    bursts_analysis['replies'] = get_stat_bursts(bursts_replies)
    print_dict_table(bursts_analysis, title='BURSTS ANALYSIS', values_unit='%', shorten_values=5)
    time_gaps_analysis = dict()
    time_gaps_analysis['all time gaps'] = get_stat_timegaps(time_gaps_all)
    time_gaps_analysis['posts time gaps'] = get_stat_timegaps(time_gaps_posts)
    time_gaps_analysis['replies time gaps'] = get_stat_timegaps(time_gaps_replies)
    print_dict_table(time_gaps_analysis, title='TIME GAPS ANALYSIS')
    print_f('time gaps analysis done')
    return time_gaps_analysis


if __name__ == '__main__':
    start = datetime.datetime.now()
    time_gaps_analysis("/opt/datasets/fitnessexchange/extracted_logfile" if len(sys.argv) <= 1 else sys.argv[1], None if len(sys.argv) <= 2 else sys.argv[2])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
