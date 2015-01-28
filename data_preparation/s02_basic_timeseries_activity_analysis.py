from __future__ import division
from util import *


def count_user(user_id):
    count_user.ids.add(int(user_id))
    return len(count_user.ids)


def bin_date(timestamp):
    # return timestamp.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    return timestamp.replace(microsecond=0, second=0, minute=0, hour=0)


def basic_timeseries_activity_analysis(log_filename, rolling_window=1):
    print_f('start basic time-series activity analysis')
    count_user.ids = set()
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    data_frame = read_log_to_df(log_filename)

    # set activity for each post to 1
    print_f('create activity col')
    data_frame['activity'] = 1

    # count users
    print_f('create users col')
    data_frame['users'] = data_frame['source'].apply(count_user)

    # normalize activity of each post by #users currently registered
    print_f('create normalized activity col')
    data_frame['normalized activity'] = data_frame['activity'] / data_frame['users']

    # compute time since last activity in network
    print_f('create time since last activity col')
    data_frame['TimeSinceLastActivity'] = data_frame['timestamp'].diff()
    # start plotting if more than X users registered

    print_f('bin data frame')
    grouped_df = data_frame.groupby(data_frame['timestamp'].map(lambda x: x.date()))
    group = grouped_df[['activity']].sum()
    group['users'] = grouped_df['users'].mean()
    group['rolling mean activity'] = pd.rolling_mean(group['activity'], rolling_window)
    print_f(group.tail())
    print_f('datapoints:', len(group))

    plt.figure()
    print_f('plot activity')
    ax = group.plot(y=['activity'])
    ax = group.plot(y=['rolling mean activity'], ax=ax, linewidth=3, grid=True)
    ax2 = group.plot(y=['users'], secondary_y=True, linewidth=3, ax=ax)
    ax2.legend(loc=1)
    ax.set_ylabel('activity\n1 = one post of one user')
    ax.right_ax.set_ylabel('#users with at least one activity')
    ax.legend(loc=2)
    plot_name = 'activity_and_users_over_time'
    plot_name = plot_name + '.png'
    plt.savefig(folder_name + plot_name, dpi=150)
    plt.close('all')

    first_two_m = data_frame.index[:1]
    print_f('drop first two months for plot:', first_two_m)
    group = grouped_df[['normalized activity']].sum()
    group['users'] = grouped_df['users'].mean()
    group['rolling mean activity'] = pd.rolling_mean(group['normalized activity'], rolling_window)
    print_f('plot norm activity activity')
    plt.clf()
    ax = group.plot(y=['normalized activity'])
    ax = group.plot(y=['rolling mean activity'], ax=ax, linewidth=3, grid=True)
    ax2 = group.plot(y=['users'], secondary_y=True, linewidth=3, ax=ax)
    ax.legend(loc=2)
    ax2.legend(loc=1)
    yrange_avg, yrange_stdd = group['normalized activity'].mean(), group['normalized activity'].std()
    ax.set_ylim([0, yrange_avg + yrange_stdd])
    ax.set_ylabel('normalized activity\nactivity/#user')
    ax.right_ax.set_ylabel('#users with at least one activity')

    plot_name = 'activity_and_users_over_time_normalized'
    plot_name = plot_name + '.png'
    plt.savefig(folder_name + plot_name, dpi=150)
    plt.close('all')
    print_f('basic time-series activity analysis done')


if __name__ == '__main__':
    start = datetime.datetime.now()
    basic_timeseries_activity_analysis("/opt/datasets/fitnessexchange/extracted_logfile" if len(sys.argv) <= 1 else sys.argv[1])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
