from __future__ import division
import math
from util import *
from collections import defaultdict


# -----------------------------------------------------------------------------------------------
class MyNet:
    def __init__(self, filename=None, create=False):
        self.filename = filename
        if self.filename is None or create is True:
            self.graph = Graph(directed=False)
            self.edge_prop = self.graph.new_edge_property("float")
            self.node_ids = self.graph.new_vertex_property("int")
            self.vertex_mapping = defaultdict(lambda: self.graph.add_vertex())
            self.filename = 'weighted_net.gt' if self.filename is None else self.filename
        else:
            print_f('load network:', self.filename)
            if not os.path.isfile(self.filename):
                print_f(self.filename, 'does not exist.')
                exit()
            else:
                self.graph = load_graph(self.filename)
                print_f('done(v:', self.graph.num_vertices(), ' e:', self.graph.num_edges(), ')')
                self.node_ids = self.graph.vertex_properties["nodeID"]
                self.edge_prop = self.graph.edge_properties["activity"]
                self.vertex_mapping = defaultdict(lambda: self.graph.add_vertex())
                print_f('create vertex mapping')
                for v in self.graph.vertices():
                    self.vertex_mapping[int(self.node_ids[v])] = v
                print_f('done')
        self.add_edge_calls = 0
        self.edges = defaultdict(lambda: defaultdict(int))

    def get_network_size(self):
        return self.graph.num_vertices(), self.graph.num_edges()

    def get_overall_activity(self):
        return sum(self.graph.edge_properties['activity'][e] for e in self.graph.edges())

    def build_network_per_line(self, df):

        for index, row in df.iterrows():
            if index % 1000 == 0:
                print "  -- Processing line {}".format(index)
            src = row['source']
            trgt = row['destination']
            ts = row['timestamp']

            if src is not "nan":
                src_v = self.vertex_mapping[src]

            if not math.isnan(trgt):
                #print trgt
                trgt_v = self.vertex_mapping[trgt]
                edge = self.graph.add_edge(src_v, trgt_v)
        for orig_id, v in self.vertex_mapping.iteritems():
            self.node_ids[v] = orig_id
        remove_parallel_edges(self.graph)
        remove_self_loops(self.graph)
        print_f('build done')
        print_f('\t=>vertices:', self.graph.num_vertices(), 'edges:', self.graph.num_edges())


    def build_network(self):
        for src, dest_dict in self.edges.iteritems():
            src_v = self.vertex_mapping[src]
            for dest, weight in dest_dict.iteritems():
                dest_v = self.vertex_mapping[dest]
                edge = self.graph.add_edge(src_v, dest_v)
                self.edge_prop[edge] = weight
        for orig_id, v in self.vertex_mapping.iteritems():
            self.node_ids[v] = orig_id
        self.edges = defaultdict(lambda: defaultdict(int))
        remove_parallel_edges(self.graph)
        print_f('build done')
        print_f('\t=>vertices:', self.graph.num_vertices(), 'edges:', self.graph.num_edges())


    def add_edges_from_df(self, df):
        for src, dest in zip(df['source'], df['destination']):
            if not np.isnan(dest) and not src == dest:
                src, dest = (src, dest) if src < dest else (dest, src)
                self.edges[src][dest] += 1

    def draw(self, filename=None, output_size=8000, largest=False, log=False):
        filename = self.filename if filename is None else filename
        if not filename.endswith('.png'):
            filename += '.png'

        if largest:
            print_f('extract largest component')
            l = label_largest_component(self.graph)
            self.graph = GraphView(self.graph, vfilt=l)
        print_f('drawing graph')
        if log:
            tmp_prop = self.graph.new_edge_property("float")
            logf = math.log
            for edge in self.graph.edges():
                tmp_prop[edge] = logf(self.edge_prop[edge])
        graph_draw(self.graph, edge_color='black', edge_pen_width=prop_to_size(self.edge_prop if not log else tmp_prop, mi=1, ma=15), output_size=(output_size, output_size),
                   output=filename)
        print_f('done')

    def get_edge_thresholds(self, thresholds=[1, 2], draw=False, output_size=10000):
        print_f('get edge th:', thresholds)
        print_f('\textract largest comp')
        l = label_largest_component(self.graph)
        laco_graph = GraphView(self.graph, vfilt=l)
        if draw:
            print_f('\tcalc positions...')
            pos = sfdp_layout(laco_graph)
            pen_width = prop_to_size(self.edge_prop, mi=1, ma=5, log=True)
            print_f('\tdrawing...')
            net_viz_folder = self.filename.rsplit('/', 1)[0] + '/net_viz'
            if not os.path.isdir(net_viz_folder):
                os.mkdir(net_viz_folder)
            net_viz_folder += '/'
        else:
            print_f('\tcalc cores...')
        largest_component_size = []
        for idx, th in enumerate(thresholds):
            print_f('\tcore', th, '(', idx / len(thresholds) * 100, '%)')
            sys.stdout.flush()
            tmp_graph = GraphView(laco_graph, efilt=lambda e: self.edge_prop[e] >= th)
            if draw:
                graph_draw(tmp_graph, pos=pos, vertex_color='blue', vertex_fill_color='blue', edge_color='black', edge_pen_width=pen_width, output_size=(output_size, output_size),
                           output=net_viz_folder + str(th).rjust(3, '0') + '_th.png')
            l = label_largest_component(tmp_graph)
            tmp_la_comp = GraphView(tmp_graph, vfilt=l)
            if draw:
                graph_draw(tmp_la_comp, pos=pos, vertex_color='blue', vertex_fill_color='blue', edge_color='black', edge_pen_width=pen_width,
                           output_size=(output_size, output_size),
                           output=net_viz_folder + str(th).rjust(3, '0') + '_th_LC.png')
            tmp_la_activity = sum(tmp_la_comp.edge_properties['activity'][e] for e in tmp_la_comp.edges())
            largest_component_size.append((tmp_la_comp.num_vertices(), tmp_la_comp.num_edges(), tmp_la_activity))
        print ''
        print_f('done')
        return largest_component_size

    def store(self, filename=None):
        filename = self.filename if filename is None else filename
        if not filename.endswith('.gt'):
            filename += '.gt'
        print_f('store network:', filename)
        self.graph.edge_properties["activity"] = self.edge_prop
        self.graph.vertex_properties["nodeID"] = self.node_ids
        self.graph.save(filename)


# -----------------------------------------------------------------------------------------------

def generate_network(log_filename, draw=False):
    print_f('start generation of network')
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    network_name = folder_name + 'net.gt'
    create_network = not os.path.isfile(network_name)
    # data_frame = pd.DataFrame(columns=['datetime', 'source', 'destination'], dtype=int)
    if create_network:
        data_frame = read_log_to_df(log_filename)
        # create graph
        network = MyNet(network_name, create=True)
        print_f('add edges from df to network cache')
        network.build_network_per_line(data_frame)
        print_f('store network')
        network.store(network_name)
    else:
        print_f("load network from:", network_name)
        network = MyNet(network_name)

    all_nodes, all_edges = network.get_network_size()
    all_activity = network.get_overall_activity()
    # network.draw(filename='net_all.png')
    # network.draw(filename='net_largest.png', largest=True, log=True)

    print_f('generation of weighted network done')
    return network_name


def generate_weighted_network(log_filename, draw=False):
    print_f('start generation of weighted network')
    folder_name = log_filename.rsplit('/', 1)[0] + '/'
    network_name = folder_name + 'weighted_net.gt'
    create_network = not os.path.isfile(network_name)
    # data_frame = pd.DataFrame(columns=['datetime', 'source', 'destination'], dtype=int)
    if create_network:
        data_frame = read_log_to_df(log_filename)
        # create graph
        network = MyNet(network_name, create=True)
        print_f('add edges from df to network cache')
        network.add_edges_from_df(data_frame)
        print_f('build network from cache')
        network.build_network()
        print_f('store network')
        network.store(network_name)
    else:
        print_f("load network from:", network_name)
        network = MyNet(network_name)

    all_nodes, all_edges = network.get_network_size()
    all_activity = network.get_overall_activity()
    # network.draw(filename='net_all.png')
    # network.draw(filename='net_largest.png', largest=True, log=True)
    thresholds = [i for i in range(1, 11)]
    largest_component_sizes = network.get_edge_thresholds(thresholds=thresholds, draw=draw)

    print_f('plot')
    over_view = ''
    result = dict()
    for norm_flag in [False, True]:
        log = False
        print largest_component_sizes
        nodes, edges, activity = zip(*largest_component_sizes)
        if norm_flag:
            norm = 1 / all_nodes * 100
            nodes = [i * norm for i in nodes]
            result['nodes'] = {key: val for key, val in zip(thresholds, nodes)}
            norm = 1 / all_edges * 100
            edges = [i * norm for i in edges]
            result['edges'] = {key: val for key, val in zip(thresholds, edges)}
            norm = 1 / all_activity * 100
            activity = [i * norm for i in activity]
            result['activity'] = {key: val for key, val in zip(thresholds, activity)}

            over_view += 'OVERVIEW'.center(41, '=') + '\n'
            over_view += 'THRESHOLD'.center(10, ' ') + '|' + 'NODES'.center(9, ' ') + '|' + 'EDGES'.center(9, ' ') + '|' + 'ACTIVITY'.center(10, ' ') + '\n'
            for idx, i in enumerate(thresholds):
                over_view += str(i).center(10, ' ') + '|' + str(str(nodes[idx])[:5] + '%').center(9, ' ') + '|' + str(str(edges[idx])[:5] + '%').center(9, ' ') + '|' + str(
                    str(activity[idx])[:5] + '%').center(10, ' ') + '\n'

        plt.clf()
        plt.figure()
        fig, ax = plt.subplots()
        print_f('plot largest component sizes')
        if log:
            ax.set_yscale('log')
        ax2 = ax.twinx()
        ax.plot(thresholds, nodes, label='nodes', c='green', linewidth=2)
        ax2.plot(thresholds, edges, label='edges', c='blue', linewidth=2)
        ax2.plot(thresholds, activity, label='activity', c='black', linewidth=2)
        ax.set_ylabel('nodes in %' if norm_flag else 'nodes')
        ax2.set_ylabel('edges/activity in %' if norm_flag else 'edges/activity')

        if log:
            ax2.set_yscale('log')
        plt.xlabel('threshold')
        ax.legend(loc=2)
        ax2.legend(loc=1)
        if norm_flag:
            ax.set_ylim([0, 100])
            ax2.set_ylim([0, 100])
        # plt.legend()
        plot_name = 'largest_comp_size' if not norm_flag else 'largest_comp_size_norm'
        plot_name = folder_name + plot_name + '.png'
        plt.grid()
        plt.savefig(plot_name, dpi=150)
        plt.close('all')
    print_f('\n', over_view)
    print_f('generation of weighted network done')
    return network_name, result


if __name__ == '__main__':
    start = datetime.datetime.now()
    generate_weighted_network("/opt/datasets/fitnessexchange/extracted_logfile" if len(sys.argv) <= 1 else sys.argv[1])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'
