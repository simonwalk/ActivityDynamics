from __future__ import division
from util import *


def calc_ratio(folder, serialized_lambda='user_df_lambda.ser', serialized_mu='user_df_mu.ser', net_filename=None, rolling_means=None, fit_curve_deg=None):
    print_f('start calculation of ratio')
    filename_lambda = folder + serialized_lambda
    filename_mu = folder + serialized_mu
    network_name = folder + net_filename
    filename = filename_mu.rsplit('_', 1)[0]

    print_f('load df from ', filename_lambda)
    lambda_df = pd.read_pickle(filename_lambda)
    print_f('load df from ', filename_mu)
    mu_df = pd.read_pickle(filename_mu)
    print mu_df.index[:2]
    print lambda_df.index[:2]
    print type(mu_df), type(lambda_df)
    print_f('calc lambda')
    print lambda_df.columns, mu_df.columns
    ratio = pd.DataFrame(columns=['ratio'], data=lambda_df['lambda'] / mu_df['mu'])
    if not rolling_means is None:
        ratio = add_rolling_means(ratio, rolling_means, 'ratio')
    if not fit_curve_deg is None:
        ratio = add_poly_fit(ratio, fit_curve_deg, 'ratio')
    # lambda_fits = [i for i in lambda_df.columns if 'poly fit' in i]
    # mu_fits = [i for i in mu_df.columns if 'poly fit' in i]
    # ratio['calc best fits'] = lambda_df[lambda_fits[0]] / mu_df[mu_fits[0]]
    #ratio['calc second best fits'] = lambda_df[lambda_fits[1]] / mu_df[mu_fits[1]]
    largest_eigenval = ''
    if not net_filename is None:
        print_f('calc largest eigenval')
        net = load_graph(net_filename)
        largest_eigenval, eigenvecs = eigenvector(net, max_iter=1000)
        ratio['largest eigenval'] = [largest_eigenval for i in xrange(len(ratio.index))]
    plot(stats=ratio, filename=filename + '_ratio.png', x_label=None if net_filename is None else 'largest eigenvalue: ' + str(largest_eigenval))
    print_f('calculation of ratio done')
    ratio.to_pickle(filename + '_ratio.ser')
    if not net_filename is None:
        ratio = ratio / largest_eigenval
        plot(stats=ratio, filename=filename + '_ratio_norm.png', x_label=None if net_filename is None else 'largest eigenvalue: ' + str(largest_eigenval))
    return ratio


if __name__ == '__main__':
    start = datetime.datetime.now()
    calc_ratio("/opt/datasets/fitnessexchange/" if len(sys.argv) <= 1 else sys.argv[1])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
