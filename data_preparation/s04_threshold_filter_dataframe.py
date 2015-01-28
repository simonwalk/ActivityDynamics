from __future__ import division
from util import *


def threshold_filter_dataframe(log_filename, net_filename=None):
    print_f('start threshold filter of dataframe')
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    network_name = folder_name + 'weighted_net.gt' if net_filename is None else net_filename
    data_frame = read_log_to_df(log_filename)
    print_f('load graph')
    graph = load_graph(network_name)
    for threshold in range(1, 10):
        print_f('filter threshold:', threshold)
        tmp_graph = GraphView(graph, efilt=lambda e: graph.edge_properties["activity"][e] >= threshold)
        nodes = {tmp_graph.vertex_properties["nodeID"][v] for v in tmp_graph.vertices()}
        data_frame['th_' + str(threshold)] = data_frame['source'].map(lambda x: x in nodes) & (
            np.isnan(data_frame['destination']) | data_frame['destination'].map(lambda x: x in nodes))
        l = label_largest_component(tmp_graph)
        tmp_la_comp = GraphView(tmp_graph, vfilt=l)
        nodes = {tmp_la_comp.vertex_properties["nodeID"][v] for v in tmp_la_comp.vertices()}
        data_frame['th_' + str(threshold)] = data_frame['source'].map(lambda x: x in nodes) & (
            np.isnan(data_frame['destination']) | data_frame['destination'].map(lambda x: x in nodes))

    print_f(data_frame.head())
    out_filename = log_filename + '_filtered_activity.ser'
    data_frame.to_pickle(out_filename)
    print_f('threshold filter of dataframe done')


if __name__ == '__main__':
    start = datetime.datetime.now()
    threshold_filter_dataframe("/opt/datasets/fitnessexchange/extracted_logfile" if len(sys.argv) <= 1 else sys.argv[1], None if len(sys.argv) <= 2 else sys.argv[2])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'