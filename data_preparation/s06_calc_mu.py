from __future__ import division
from util import *


def calc_mu(folder, serialized_posts='user_df_posts.ser', serialized_replies='user_df_replies.ser', rolling_window=1, filter_inactive=True, drop_last_m=False, rolling_means=None,
            fit_curve_deg=None):
    print_f('start calculation of mu')
    filename_posts = folder + serialized_posts
    filename_replies = folder + serialized_replies

    print_f('load df from ', filename_posts)
    posts = pd.read_pickle(filename_posts)
    print_f('load df from ', filename_replies)
    replies = pd.read_pickle(filename_replies)
    print_f('cols dataframes:', len(posts.columns), len(replies.columns))
    if len(posts.columns) != len(replies.columns):
        print_f('Not equal amount of cols in', filename_posts, 'and', filename_replies)
    filename = filename_posts.rsplit('.', 1)[0].rsplit('_', 1)[0]
    print_f('df contains', len(posts), 'timestamps and', len(posts.columns), 'users')
    posts.fillna(value=0, inplace=True)
    replies.fillna(value=0, inplace=True)
    if drop_last_m:
        last_m = posts.index[-1]
        print_f('drop last month:', last_m)
        posts.drop(last_m, inplace=True)
        replies.drop(last_m, inplace=True)

    if rolling_window > 1:
        print_f('rolling sum with window-size:', rolling_window)
        mod_posts = pd.rolling_sum(posts, window=rolling_window, axis=0)
        mod_replies = pd.rolling_sum(replies, window=rolling_window, axis=0)
    else:
        mod_posts = posts.copy()
        mod_replies = replies.copy()
    print_f('filter inactive users to nan')
    mod_posts = mod_posts.applymap(func=lambda x: np.nan if x == 0 else x)
    mod_replies = mod_replies.applymap(func=lambda x: np.nan if x == 0 else x)

    print_f('calc active stat posts')
    norm_posts = 1 / len(mod_posts.columns) * 100
    active = pd.DataFrame(columns=['posts'], data=mod_posts.apply(func=lambda x: np.sum(np.invert(np.isnan(x))) * norm_posts, axis=1))

    print_f('calc active stat replies')
    norm_replies = 1 / len(mod_replies.columns) * 100
    active['replies'] = mod_replies.apply(func=lambda x: np.sum(np.invert(np.isnan(x))) * norm_replies, axis=1)

    plt.clf()
    fig, ax = plt.subplots()
    active.plot(ax=ax, linewidth=3)
    ax.set_ylabel('active users in %')
    plt.savefig(filename=filename + '_' + str(rolling_window) + 'm_active_users.png', dpi=150, ylabel='percentage of active users')
    plt.close('all')

    print_f('calc row sums')
    posts_rsum = posts.sum(axis=1)+1
    replies_rsum = replies.sum(axis=1)+1

    print_f('calc active users')
    active_users_posts = mod_posts.apply(func=lambda x: np.invert(np.isnan(x)))+1
    active_users_replies = mod_replies.apply(func=lambda x: np.invert(np.isnan(x)))+1
    active_users = active_users_posts | active_users_replies
    active_users = active_users.apply(func=lambda x: np.sum(x), axis=1) + 1

    #print_f("posts_rsum")
    #print posts_rsum
    #print_f("replies_rsum")
    #print replies_rsum
    print_f("active_users")
    print active_users
    print_f('calc ac')
    ac_df = pd.DataFrame(columns=['alpha critical'], data=(posts_rsum / replies_rsum))
    print ac_df
    if not rolling_means is None:
        ac_df = add_rolling_means(ac_df, rolling_means, 'alpha critical')
    if not fit_curve_deg is None:
        ac_df = add_poly_fit(ac_df, fit_curve_deg, 'alpha critical')
    plot(stats=ac_df, filename=filename + '_' + str(rolling_window) + 'm_ac.png', custom_labels='alpha critical')

    print_f('calc q')
    q_df = pd.DataFrame(columns=['q'], data=(replies_rsum / posts_rsum) / active_users)
    if not rolling_means is None:
        q_df = add_rolling_means(q_df, rolling_means, 'q')
    if not fit_curve_deg is None:
        q_df = add_poly_fit(q_df, fit_curve_deg, 'q')
    plot(stats=q_df, filename=filename + '_' + str(rolling_window) + 'm_q.png')

    print_f('calc mu')
    print 'ac cols:', ac_df.columns
    print 'q cols:', q_df.columns
    mu_df = pd.DataFrame(columns=['mu'], data=q_df['q'] / ac_df['alpha critical'])
    tails = 3
    print 'posts rsum', posts_rsum.tail(tails)
    print 'replies rsum', replies_rsum.tail(tails)
    print 'q\n', q_df.tail(tails)
    print 'ac\n', ac_df.tail(tails)
    print 'mu\n', mu_df.tail(tails)
    if not rolling_means is None:
        mu_df = add_rolling_means(mu_df, rolling_means, 'mu')
    if not fit_curve_deg is None:
        mu_df = add_poly_fit(mu_df, fit_curve_deg, 'mu')
    plot(stats=mu_df, filename=filename + '_' + str(rolling_window) + 'm_mu.png', custom_labels=['$\\mu$'])
    mu_df.to_pickle(filename + '_mu.ser')
    print_f('calculation of mu done')
    return mu_df


if __name__ == '__main__':
    start = datetime.datetime.now()
    calc_mu("/opt/datasets/fitnessexchange/" if len(sys.argv) <= 1 else sys.argv[1])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
