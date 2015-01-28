from __future__ import division
from util import *


def main():
    test = False
    stats = False
    input_filename = sys.argv[1]
    output_filename = input_filename.rsplit('.', 1)[0] + '.csv'
    print_f("read df:", input_filename)
    data_frame = pd.read_pickle(input_filename)
    print 'index len:', len(data_frame), '/ cols:', len(data_frame.columns)
    if test:
        data_frame = data_frame[data_frame.columns[:3]]
    print_f("write df:", output_filename)
    data_frame.to_csv(output_filename, sep=',', na_rep="0")
    if stats:
        output_filename = input_filename.rsplit('.', 1)[0] + '_stat.csv'
        print_f('calc avg')
        stat_df = pd.DataFrame(columns=['avg'], data=data_frame.mean(axis=1))
        print_f('calc median')
        stat_df['median'] = data_frame.median(axis=1)
        print_f("write stats df:", output_filename)
        stat_df.to_csv(output_filename, sep=',', na_rep="0")
    print_f("done")


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
