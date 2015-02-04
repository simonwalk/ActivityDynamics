from __future__ import division
from util import *


def bin_date(timestamp):
    # return timestamp.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    return timestamp.replace(microsecond=0, second=0, minute=0, hour=0)


def extract_binned_posts_replies(log_filename, net_filename=None, core=0, sample_size=-1):
    print_f('start extraction of binned posts & replies')
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    network_name = folder_name + 'weighted_net.gt' if net_filename is None else net_filename

    data_frame = read_log_to_df(log_filename)
    print_f('simplify timestamps')
    data_frame['timestamp'] = data_frame['timestamp'].map(lambda x: x.date().replace(day=1))
    core = 0
    if core > 0:
        graph = read_graph(network_name, activity_threshold=core, largest_component=False)

    if core > 0:
        print_f('create set of user-ids of network')
        user_ids = {int(graph.vp["nodeID"][v]) for v in graph.vertices()}
        print_f('core', core, 'contains:', len(user_ids), 'users')
    else:
        print_f('create set of user-ids of dataframe')
        user_ids = set(data_frame['source'].astype('int')) | set(data_frame['destination'][data_frame['destination'].notnull()].astype('int'))
        print_f('dataset contains', len(user_ids), 'users')

    #if sample_size == -1 or sample_size >= len(user_ids):
    #    user_ids = set(user_ids)
    #else:
    #    print_f('sample', sample_size, 'users')
    #    user_ids = set(random.sample(user_ids, sample_size))
    print_f('calc time-span...')
    min_date, max_date = np.min(data_frame['timestamp']).replace(day=1), np.max(data_frame['timestamp']).replace(day=1)
    dates_index = []
    date_stamp = min_date
    print 'min date:', min_date
    print 'max date:', max_date
    print_f('create time-span index range...')
    while date_stamp <= max_date:
        dates_index.append(date_stamp)
        try:
            date_stamp = date_stamp.replace(month=date_stamp.month + 1)
        except:
            date_stamp = date_stamp.replace(year=date_stamp.year + 1, month=1)
    print_f('done')
    print 'range: #', len(dates_index)
    print dates_index
    print_f('create empty dataframes')
    users_df_posts = pd.DataFrame(data=[[np.nan] * len(user_ids)] * len(dates_index), index=dates_index, columns=sorted(user_ids), dtype=np.float)
    users_df_replies = pd.DataFrame(data=[[np.nan] * len(user_ids)] * len(dates_index), index=dates_index, columns=sorted(user_ids), dtype=np.float)
    print_f('done')
    print_f('empty frames:')
    print 'index len:', len(users_df_posts), '(timespan len:', len(dates_index), ')'
    print 'cols:', len(users_df_posts.columns), '(users:', len(user_ids), ')'
    print_f('bin source')
    src_bin_df = data_frame.groupby('source')
    print_f('done')
    print_f('start processing...')
    percent = -1
    len_users = len(user_ids)
    process_times = []
    prog = 0
    start = datetime.datetime.now()

    for user, gdf in src_bin_df:
        user = int(user)
        if user in user_ids:
            prog += 1
            # count posts,replies per bin
            user_col_posts = users_df_posts[user]
            user_col_replies = users_df_replies[user]
            for bin_timestamp, tmp_df in gdf.groupby('timestamp'):
                if not len(tmp_df) == 0:
                    # count = #not nan
                    replies = tmp_df.destination.count()
                    posts = len(tmp_df) - replies
                    user_col_posts[bin_timestamp] = posts
                    user_col_replies[bin_timestamp] = replies

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
    users_df_posts.to_pickle(folder_name + 'user_df_posts.ser')
    users_df_replies.to_pickle(folder_name + 'user_df_replies.ser')
    print_f('extraction of binned posts & replies done')


if __name__ == '__main__':
    start = datetime.datetime.now()
    extract_binned_posts_replies("/opt/datasets/fitnessexchange/extracted_logfile" if len(sys.argv) <= 1 else sys.argv[1], None if len(sys.argv) <= 2 else sys.argv[2])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
