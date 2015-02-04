from __future__ import division
from util import *


def bin_date(timestamp):
    # return timestamp.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    return timestamp.replace(microsecond=0, second=0, minute=0, hour=0)


def core_activity_analysis(log_filename, net_filename='weighted_net.gt', core=0, sample_size=-1):
    print_f('start core activity analysis')
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    network_name = folder_name + net_filename

    data_frame = read_log_to_df(log_filename)
    print_f('simplify timestamps')
    data_frame['timestamp'] = data_frame['timestamp'].map(lambda x: x.date().replace(day=1))
    if core > 0:
        graph = read_graph(network_name, activity_threshold=core, largest_component=True)

    user_ids = {int(i) for i in data_frame['source']}
    print_f('dataset contains', len(user_ids), 'users')
    num_all_users = len(user_ids)

    # filter core users
    print_f('create set of user-ids')
    core=0
    print core
    #if core > 0:
    #    user_ids = {int(graph.vertex_properties["nodeID"][v]) for v in graph.vertices()}
    #    core_users_percentage = len(user_ids) / num_all_users * 100
    #    print_f('core', core, 'contains:', len(user_ids), 'users', '(', core_users_percentage, '% )')
    #else:
    core_users_percentage = 100

    # sampling users
    if sample_size == -1 or sample_size >= len(user_ids):
        user_ids = set(user_ids)
    else:
        print_f('sample', sample_size, 'users')
        user_ids = set(random.sample(user_ids, sample_size))

    print_f('filter users by source...')
    data_frame = data_frame[data_frame['source'].map(lambda x: x in user_ids)]

    print_f('filter posts...')
    posts = data_frame[np.isnan(data_frame['destination'])]

    print_f('filter replies...')
    replies = data_frame[data_frame['destination'].map(lambda x: x in user_ids)]

    print_f('group posts...')
    posts = posts.groupby('timestamp').size()
    # posts.sort('timestamp', inplace=True)

    print_f('group replies...')
    replies = replies.groupby('timestamp').size()
    # replies.sort('timestamp', inplace=True)


    print_f('create joined dataframe')
    print 'post type:', type(posts), '\n', posts.head(3)
    print 'replies type:', type(replies), '\n', replies.head(3)
    data_frame = pd.DataFrame(columns=['posts'], data=posts)
    # data_frame['posts'] = posts
    data_frame['replies'] = replies
    data_frame.sort(inplace=True)
    last_m = data_frame.index[-1]
    print_f('drop last month:', last_m)
    data_frame.drop(last_m, inplace=True)
    print 'final df:\n'
    print data_frame.head(3)
    print data_frame.tail(3)

    print_f('plotting...')
    plot(stats=data_frame, filename=folder_name + 'core_activity.png', x_label='#users: ' + str(len(user_ids)) + ' ( ' + str(core_users_percentage) + '% )',
         second_y_axis=data_frame.columns[-1])
    print_f('core activity analysis done')


if __name__ == '__main__':
    start = datetime.datetime.now()
    core_activity_analysis("/opt/datasets/fitnessexchange/extracted_logfile" if len(sys.argv) <= 1 else sys.argv[1], None if len(sys.argv) <= 2 else sys.argv[2])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
