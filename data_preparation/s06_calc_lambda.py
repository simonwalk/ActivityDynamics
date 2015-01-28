from __future__ import division
from util import *


def calc_lambda(folder, serialized_posts='user_df_posts.ser', rolling_window=1, filter_inactive=True, drop_last_m=True, random_users=0, rolling_means=None, fit_curve_deg=None):
    print_f('start calculation of lambda')
    serialized_posts = folder + serialized_posts

    print_f('load df from ', serialized_posts)
    posts = pd.read_pickle(serialized_posts)
    serialized_posts = serialized_posts.rsplit('_', 1)[0]
    print_f('df contains', len(posts), 'timestamps and', len(posts.columns), 'users')
    posts.fillna(value=0, inplace=True)
    if drop_last_m:
        last_m = posts.index[-1]
        print_f('drop last month:', last_m)
        posts.drop(last_m, inplace=True)
    if random_users != 0:
        rand_user_ids = list(set(posts.columns)) if random_users == -1 else list(random.sample(set(posts.columns), random_users))
    if rolling_window > 1:
        print_f('rolling sum with window-size:', rolling_window)
        mod_posts = pd.rolling_sum(posts, window=rolling_window, axis=0)
    else:
        mod_posts = posts.copy()
    mod_posts += 1
    if filter_inactive:
        print_f('set inactive users to nan')
        mod_posts = mod_posts.applymap(func=lambda x: np.nan if x == 1 else x)
    print_f('calc lambdas')
    lambda_df = pd.rolling_apply(mod_posts, window=rolling_window + 1, func=lambda x: (x[0]+1 - x[-1]+1) / x[0]+1, min_periods=rolling_window + 1)
    print_f('done')
    print_f('save df')
    print_f('calc avg')
    stat_df = pd.DataFrame(columns=['lambda'], data=lambda_df.mean(axis=1))
    print_f('calc median')
    stat_df['median'] = lambda_df.median(axis=1)
    if not rolling_means is None:
        stat_df = add_rolling_means(stat_df, rolling_means, col_name='lambda')
    if not fit_curve_deg is None:
        stat_df = add_poly_fit(stat_df, fit_curve_deg, col_name='lambda')
    sample_users = None
    if random_users != 0:
        sample_users = lambda_df[rand_user_ids]
    plot(stats=stat_df, users=sample_users, filename=serialized_posts + '_' + str(rolling_window) + 'm_lambda.png', logy=False,
         custom_labels=['$\\lambda$ mean', '$\\lambda$ median'])
    print_f('calculation of lambda done')
    stat_df.to_pickle(serialized_posts + '_lambda.ser')
    return stat_df


if __name__ == '__main__':
    start = datetime.datetime.now()
    calc_lambda("/opt/datasets/fitnessexchange/" if len(sys.argv) <= 1 else sys.argv[1], None if len(sys.argv) <= 2 else sys.argv[2])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
