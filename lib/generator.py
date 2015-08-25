from __future__ import division

__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

from lib.util import *
from random import randint, uniform, random
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from random import shuffle
from scipy.stats import poisson
import matplotlib
from dateutil.relativedelta import relativedelta


# Generator Class works with GraphTool generators, as they provide more functionality than NetworkX Generators
class Generator():
    # init generator
    def __init__(self, graph_name, num_nodes=5000, directed=False):
        self.graph = Graph(directed=directed)
        self.directed = directed
        self.num_nodes = num_nodes
        self.graph_name = graph_name


    # create network meta function that calls the corresponding functions of the generator class
    def create_network(self, generator_function, generator_arguments=[], draw_graph=False, draw_init_number=0,
                       draw_ev=False, debug_msg="Creating Network", node_weight_min=0.0, node_weight_max=0.1):
        self.debug_msg(debug_msg)
        generator_function(*generator_arguments)
        self.add_node_weights(node_weight_min, node_weight_max)
        self.calc_eigenvalues(max(int(self.graph.num_vertices()/2)-1, 1))
        self.collect_colors()
        self.calc_vertex_properties()
        if draw_graph:
            self.draw_graph(draw_init_number)
        if draw_ev:
            self.prepare_eigenvalues()
            self.plot_eigenvalues()
        self.store_graph(0)


    # start creating blockmodel graph
    def create_blockmodel_graph(self, blocks=7, connectivity=10, model="blockmodel-traditional"):
        def corr(a, b):
            if a == b:
                return 0.999
            else:
                return 0.001

        self.debug_msg("Starting to create Blockmodel Graph with {} nodes and {} blocks".format(self.num_nodes, blocks))

        self.graph, vertex_colors = random_graph(self.num_nodes, lambda: poisson(connectivity), directed=False,
                                                 model=model, block_membership=lambda: randint(1, blocks),
                                                 vertex_corr=corr)
        self.graph.vertex_properties["colorsComm"] = vertex_colors

    def create_fully_connected_graph(self, size=1000, directed=False, self_edges=False):
        self.create_stochastic_blockmodel_graph(blocks=1, size=size, directed=directed, self_edges=self_edges,
                                                self_block_connectivity=1.0, other_block_connectivity=1.0)

    def create_sbm_lined_up_matrix(self, blocks=10, self_block_connectivity=[0.9], other_block_connectivity=[0.1]):
        connectivity_matrix = []
        blocks_range = range(blocks)
        for idx in blocks_range:
            row = []
            outer_prob = other_block_connectivity[idx % len(other_block_connectivity)]
            inner_prob = self_block_connectivity[idx % len(self_block_connectivity)]
            for jdx in blocks_range:
                if idx != jdx:
                    row.append(outer_prob / pow(abs(idx - jdx), 2))
                else:
                    row.append(inner_prob)
            connectivity_matrix.append(row)
        return connectivity_matrix

    # scale = None
    # scale = relative
    # scale = absolute
    def create_stochastic_blockmodel_graph(self, blocks=10, size=[100], self_block_connectivity=[0.9],
                                           other_block_connectivity=[0.1], connectivity_matrix=None,
                                           directed=False, self_edges=False, power_exp=None, scale=None):
        size = size if isinstance(size, list) else [size]
        self_block_connectivity = self_block_connectivity if isinstance(self_block_connectivity, list)\
            else [self_block_connectivity]
        other_block_connectivity = other_block_connectivity if isinstance(other_block_connectivity, list)\
            else [other_block_connectivity]
        num_nodes = sum([size[i % len(size)] for i in xrange(blocks)])
        if power_exp is None:
            self.debug_msg("Starting to create Stochastic Blockmodel Graph with {} nodes and {} "
                           "blocks".format(num_nodes, blocks))
        else:
            self.debug_msg("Starting to create degree-corrected (alpha=" + str(power_exp) + ") Stochastic Blockmodel "
                                                                                            "Graph with {} nodes and {}"
                                                                                            " blocks".format(num_nodes,
                                                                                                             blocks))
        self.debug_msg('convert/transform probabilities')
        blocks_range = range(blocks)
        block_sizes = np.array([size[i % len(size)] for i in blocks_range])
        # create connectivity matrix of self- and other-block-connectivity
        if connectivity_matrix is None:
            connectivity_matrix = []
            self.debug_msg('inner conn: ' + str(self_block_connectivity) + '\tother conn: ' + str(other_block_connectivity))
            for idx in blocks_range:
                row = []
                for jdx in blocks_range:
                    if idx == jdx:
                        row.append(self_block_connectivity[idx % len(self_block_connectivity)])
                    else:
                        if not scale is None:
                            prob = other_block_connectivity[idx % len(other_block_connectivity)] / (num_nodes - block_sizes[idx]) * block_sizes[jdx]
                            if directed:
                                row.append(prob)
                            else:
                                row.append(prob / 2)
                        else:
                            row.append(other_block_connectivity[idx % len(other_block_connectivity)])
                connectivity_matrix.append(row)
        # convert con-matrix to np.array
        if not connectivity_matrix is None and isinstance(connectivity_matrix, np.matrix):
            connectivity_matrix = np.asarray(connectivity_matrix)
        # convert con-matrix to np.array
        if not connectivity_matrix is None and not isinstance(connectivity_matrix, np.ndarray):
            connectivity_matrix = np.array(connectivity_matrix)
        self.debug_msg('conn mat')
        #print_matrix(connectivity_matrix)
        if scale == 'relative' or scale == 'absolute':
            new_connectivity_matrix = []
            all_nodes = np.sum(block_sizes)
            for i in blocks_range:
                connectivity_row = connectivity_matrix[i, :] if not connectivity_matrix is None else None
                nodes_in_src_block = block_sizes[i]
                multp = 1 if scale == 'absolute' else (nodes_in_src_block * (nodes_in_src_block - 1))
                row_prob = [(connectivity_row[idx] * multp) / (nodes_in_src_block * (nodes_in_block - 1)) for idx, nodes_in_block in enumerate(block_sizes)]
                new_connectivity_matrix.append(np.array(row_prob))
            connectivity_matrix = np.array(new_connectivity_matrix)
            self.debug_msg(scale + ' scaled conn mat:')
            #print_matrix(connectivity_matrix)
        # create nodes and store corresponding block-id
        self.debug_msg('insert nodes')
        graph = Graph(directed=directed)
        vertex_to_block = []
        appender = vertex_to_block.append
        colors = graph.new_vertex_property("float")
        for i in xrange(blocks):
            block_size = size[i % len(size)]
            for j in xrange(block_size):
                appender((graph.add_vertex(), i))
                node = vertex_to_block[-1][0]
                colors[node] = i
        # create edges
        get_rand = np.random.random
        add_edge = graph.add_edge
        self.debug_msg('create edge probs')
        degree_probs = defaultdict(lambda: dict())
        for vertex, block_id in vertex_to_block:
            if power_exp is None:
                # degree_probs[block_id][vertex] = np.random.random()
                degree_probs[block_id][vertex] = 1
            else:
                # degree_probs[block_id][vertex] = 1 - pow(((pow(1, power_exp + 1) - pow(0, power_exp + 1)) * np.random.random()) + pow(0, power_exp + 1), (1 / (power_exp + 1)))
                degree_probs[block_id][vertex] = math.exp(power_exp * np.random.random())

        tmp = dict()
        self.debug_msg('normalize edge probs')
        all_prop = []
        for block_id, node_to_prop in degree_probs.iteritems():
            sum_of_block_norm = 1 / sum(node_to_prop.values())
            tmp[block_id] = {key: val * sum_of_block_norm for key, val in node_to_prop.iteritems()}
            all_prop.append(tmp[block_id].values())
        degree_probs = tmp
        plt.clf()
        plt.hist(all_prop, bins=15)
        plt.savefig("prop_dist.png")

        self.debug_msg('count edges between blocks')
        edges_between_blocks = defaultdict(lambda: defaultdict(int))
        for idx, (src_node, src_block) in enumerate(vertex_to_block):
            conn_mat_row = connectivity_matrix[src_block, :]
            for dest_node, dest_block in vertex_to_block:
                if get_rand() < conn_mat_row[dest_block]:
                    edges_between_blocks[src_block][dest_block] += 1

        self.debug_msg('create edges')
        edges_to_add = []
        for src_block, dest_dict in edges_between_blocks.iteritems():
            self.debug_msg(' -- Processing Block {}. Creating links to: {}'.format(src_block, dest_dict))
            for dest_block, num_edges in dest_dict.iteritems():
                self.debug_msg('   ++ adding {} edges to {}'.format(num_edges, dest_block))
                for i in xrange(num_edges):
                    # find src node
                    prob = np.random.random()
                    prob_sum = 0
                    src_node = None
                    for vertex, v_prob in degree_probs[src_block].iteritems():
                        prob_sum += v_prob
                        if prob_sum >= prob:
                            src_node = vertex
                            break
                    # find dest node
                    prob = np.random.random()
                    prob_sum = 0
                    dest_node = None
                    for vertex, v_prob in degree_probs[dest_block].iteritems():
                        prob_sum += v_prob
                        if prob_sum >= prob:
                            dest_node = vertex
                            break
                    if src_node is None or dest_node is None:
                        print 'Error selecting node:', src_node, dest_node
                    if graph.edge(src_node, dest_node) is None:
                        if self_edges or not src_node == dest_node:
                            add_edge(src_node, dest_node)
        self.debug_msg('remove parallel edges')
        remove_parallel_edges(graph)
        self.graph = graph
        self.graph.vertex_properties["colorsComm"] = colors


    def create_preferential_attachment(self, communities=10):
        self.graph = price_network(self.num_nodes, directed=False, c=0, gamma=1, m=1)
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 1000, communities)


    # add node to graph and check if node is in node_dict
    def add_node(self, id, pmap, node_dict, name_prop=""):
        if id in node_dict.keys():
            return self.graph.vertex(node_dict[id])
        else:
            v = self.graph.add_vertex()
            if name_prop != "":
                name_prop[v] = id
            pmap[v] = int(v)
            node_dict[id] = int(v)
            return v


    def load_smw_collab_network(self, fname, communities=10):
        self.debug_msg("Creating Graph")
        id_prop = self.graph.new_vertex_property("int")
        name_prop = self.graph.new_vertex_property("string")
        f = open(config.graph_source_dir + fname, "rb")
        id_to_index = {}
        for idx, line in enumerate(f):
            if idx % 1000 == 0:
                self.debug_msg("--> parsing line %d" % idx)
            split_line = line.strip("\n").split("\t")
            source_v = self.add_node(split_line[0], id_prop, id_to_index, name_prop)
            if split_line[1] != "":
                target_v = self.add_node(split_line[1], id_prop, id_to_index, name_prop)
                self.graph.add_edge(source_v, target_v)
        self.graph.vertex_properties["label"] = name_prop
        self.debug_msg("Detecting Communities")
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 1000, communities)
        remove_self_loops(self.graph)
        remove_parallel_edges(self.graph)


    def reduce_to_largest_component(self):
        self.debug_msg("Reducing graph to largest connected component!")
        l = label_largest_component(self.graph)
        self.graph.set_vertex_filter(l)
        self.graph.purge_vertices(in_place=True)
        self.ratio_ones = [1.0] * self.graph.num_vertices()


    def clear_all_filters(self):
        self.graph.clear_filters()
        self.num_vertices = self.graph.num_vertices()
        self.ratio_ones = [1.0] * self.num_vertices


    def increment_neighbours(self, vertices, b):
        for n in vertices:
            b[int(n)] += 1


    def load_facebook_graph(self, fname="facebook_combined.txt", communities=10):
        self.debug_msg("Creating Graph")
        id_prop = self.graph.new_vertex_property("int")
        f = open(config.graph_source_dir + fname, "rb")
        id_to_index = {}
        for idx, line in enumerate(f):
            if idx % 1000 == 0:
                self.debug_msg("--> parsing line %d" % idx)
            split_line = line.strip().split()
            source_v = self.add_node(split_line[0], id_prop, id_to_index)
            target_v = self.add_node(split_line[1], id_prop, id_to_index)
            self.graph.add_edge(source_v, target_v)
        self.graph.vertex_properties["id"] = id_prop
        self.debug_msg("Detecting Communities")
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 1000, communities)


    # start creating random graph
    # NOTE:
    # If min_degree is too small, graph will be disconnected and hence, consist of many smaller graphs!
    def create_random_graph(self, min_degree=2, max_degree=40, model="probabilistic", communities=10):
        def sample_k(min, max):
            accept = False
            while not accept:
                k = randint(min, max + 1)
                accept = random() < 1.0 / k
            return k
        self.graph = random_graph(self.num_nodes, lambda: sample_k(min_degree, max_degree), model=model,
                                  vertex_corr=lambda i, k: 1.0 / (1 + abs(i - k)), directed=self.directed, n_iter=100)
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 10000, max_degree / communities)


    # start loading  graph
    def create_karate_graph(self):
        self.graph = collection.data["karate"]
        # Removing descriptions and readme, as they screw with the GML parser of networkx!
        self.graph.graph_properties['description'] = ''
        self.graph.graph_properties['readme'] = ''
        # Calculating Colors and updating members
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 10000, 2)
        self.directed = self.graph.is_directed()
        self.num_nodes = self.graph.num_vertices()
        remove_parallel_edges(self.graph)

    def reduce_network_to_epoch(self, cur_epoch, start_date):
        self.debug_msg("Reducing network...")
        timedelta = cur_epoch + 1
        temp_epoch = start_date + relativedelta(months=timedelta)
        end_day = datetime.date(temp_epoch.year, temp_epoch.month, 1) - datetime.timedelta(days=1)
        self.debug_msg(" -> Getting network epoch from " + str(start_date) + " to " + str(end_day))
        self.graph.clear_filters()
        bool_map = self.graph.new_vertex_property("bool")
        for v in self.graph.vertices():
            if self.graph.vertex_properties["firstActivity"][v] > end_day:
                bool_map[v] = 0
            else:
                bool_map[v] = 1
        self.graph.set_vertex_filter(bool_map)
        #self.graph.purge_vertices()
        self.debug_msg("Reduced network to " + str(self.graph.num_vertices()) + " vertices with " +
                       str(self.graph.num_edges()) + " edges.")


    # collecting colors for plots
    def collect_colors(self, alpha=0.75):
        self.debug_msg("Collecting Colors for Graphs")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.graph.num_vertices())
        cmap = plt.get_cmap('gist_rainbow')
        norma = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        camap = plt.get_cmap("Blues")
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        ma = matplotlib.cm.ScalarMappable(norm=norma, cmap=camap)
        clist = self.graph.new_vertex_property("vector<float>")
        calist = self.graph.new_vertex_property("vector<float>")
        for x in xrange(self.graph.num_vertices()):
            # color for node id
            l = list(m.to_rgba(x))
            l[3] = alpha
            node = self.graph.vertex(x)
            clist[node] = l
            # color for activity / weight of node
            weight = self.graph.vertex_properties["activity"][node]
            la = list(ma.to_rgba(weight))
            la[3] = alpha
            calist[node] = la
        self.graph.vertex_properties["colorsMapping"] = clist
        self.graph.vertex_properties["colorsActivity"] = calist
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 1000, 10)


    # calculating vertex properties (yet mostly unused)
    def calc_vertex_properties(self, max_iter_ev=1000, max_iter_hits=1000):
        self.debug_msg("\x1b[33m  ++ Calculating PageRank\x1b[00m")
        pr = pagerank(self.graph)
        self.graph.vertex_properties["pagerank"] = pr
        self.debug_msg("\x1b[33m  ++ Calculating Clustering Coefficient\x1b[00m")
        clustering = local_clustering(self.graph)
        self.graph.vertex_properties["clustercoeff"] = clustering
        self.debug_msg("\x1b[33m  ++ Calculating Eigenvector Centrality\x1b[00m")
        ev, ev_centrality = eigenvector(self.graph, weight=None, max_iter=max_iter_ev)
        self.graph.vertex_properties["evcentrality"] = ev_centrality
        self.debug_msg("\x1b[33m  ++ Calculating HITS\x1b[00m")
        eig, authorities, hubs = hits(self.graph, weight=None, max_iter=max_iter_hits)
        self.graph.vertex_properties["authorities"] = authorities
        self.graph.vertex_properties["hubs"] = hubs

        self.debug_msg("\x1b[33m  ++ Calculating Degree Property Map\x1b[00m")
        degree = self.graph.degree_property_map("total")
        self.graph.vertex_properties["degree"] = degree


    def draw_specific_graph(self, colors, color_type_in_outfname, output_size, label_color, edge_color, mi, ma, labels,
                            run, appendix, file_format, label_pos=0, pos=None, v_size_prop_map=None):
        if pos is None:
            try:
                pos = self.graph.vertex_properties["pos"]
            except:
                self.debug_msg("  --> Calculating SFDP layout positions!")
                pos = sfdp_layout(self.graph, max_iter=3, verbose=False)
                self.graph.vertex_properties["pos"] = pos
                self.debug_msg("  --> Done!")
        if v_size_prop_map is None:
            try:
                v_size_prop_map = self.graph.vertex_properties["activity"]
            except:
                self.add_node_weights(0.0, 0.1)
                v_size_prop_map = self.graph.vertex_properties["activity"]
        graph_draw(self.graph, vertex_fill_color=colors, edge_color=edge_color, output_size=(output_size, output_size),
                   vertex_text_color=label_color, pos=pos, vertex_size=(prop_to_size(v_size_prop_map, mi=mi, ma=ma)),
                   vertex_text=labels, vertex_text_position=label_pos,
                   output=config.graph_dir + "{}_{}_run_{}{}.{}".format(self.graph_name, color_type_in_outfname, run,
                                                                        appendix, file_format))


    # plot graph to file
    def draw_graph(self, run=0, min_nsize=None, max_nsize=None, size_property=None, file_format="png",
                   output_size=4000, appendix="", label_color="orange", draw_labels=False):
        self.debug_msg("Drawing {}".format(file_format))
        if size_property == "degree":
            size_map = self.graph.new_vertex_property('float')
            for v in self.graph.vertices():
                size_map[v] = v.out_degree() + v.in_degree()
        if not (isinstance(size_property, int) or isinstance(size_property, float), isinstance(size_property, str)):
            size_map = size_property
        if min_nsize is None or max_nsize is None:
            val = math.sqrt(self.graph.num_vertices()) / self.graph.num_vertices() * (output_size / 4)
            mi = val if min_nsize is None else min_nsize
            ma = val * 2 if max_nsize is None else max_nsize
        if draw_labels:
            try:
                labels = self.graph.vertex_properties["label"]
            except:
                ls = self.graph.new_vertex_property("int")
                for ndx, n in enumerate(self.graph.vertices()):
                    ls[n] = str(ndx)
                self.graph.vertex_properties["label"] = ls
                labels = self.graph.vertex_properties["label"]
        else:
            labels = self.graph.new_vertex_property("string")
        if not size_property is None:
            try:
                self.draw_specific_graph(self.graph.vertex_properties["colorsComm"], "communities", output_size,
                                         label_color, "black", mi, ma, labels, run, appendix, file_format,
                                         label_pos=0, v_size_prop_map=size_map)
            except Exception as e:
                self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")
        else:
            try:
                self.draw_specific_graph(self.graph.vertex_properties["colorsComm"], "communities", output_size,
                                         label_color, "black", mi, ma, labels, run, appendix, file_format, label_pos=0)
            except Exception as e:
                self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")

            try:
                self.draw_specific_graph(self.graph.vertex_properties["colorsMapping"], "mapping", output_size,
                                         label_color, "black", mi, ma, labels, run, appendix, file_format, label_pos=0)
            except Exception as e:
                self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")

            try:
                self.draw_specific_graph(self.graph.vertex_properties["colorsActivity"], "activity", output_size,
                                         label_color, "black", mi, ma, labels, run, appendix, file_format, label_pos=0)
            except Exception as e:
                self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")


    # store graph
    def store_graph(self, run):
        self.debug_msg("Storing Graph")
        path = config.graph_binary_dir + "/GT/{}/".format(self.graph_name)
        try:
            if not os.path.exists(path):
                self.debug_msg("Created folder: {}".format(path))
                os.makedirs(path)
        except Exception as e:
            self.debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))
        self.graph.save(path + "{}_run_{}.gt".format(self.graph_name, run))


    # load graph from file
    def load_graph(self, fn):
        self.debug_msg("Loading GT")
        self.graph = load_graph(config.graph_binary_dir + "GT/" + self.graph_name + "/{}.gt".format(fn))
        self.directed = self.graph.is_directed()
        self.num_nodes = self.graph.num_vertices()
        self.debug_msg("Graph loaded with {} nodes and {} edges".format(self.graph.num_vertices(),
                                                                        self.graph.num_edges()))


    # load graph from file
    def load_smw(self, fn):
        self.debug_msg("Loading GT")
        self.graph = load_graph(config.graph_binary_dir + "GT/" + self.graph_name + "/{}.gt".format(fn))
        self.directed = self.graph.is_directed()
        self.num_nodes = self.graph.num_vertices()
        self.debug_msg("Graph loaded with {} nodes and {} edges".format(self.graph.num_vertices(),
                                                                        self.graph.num_edges()))

    def track_weight_initialization(self):
        bools = self.graph.new_vertex_property("bool")
        for v in self.graph.vertices():
            bools[v] = False
        self.graph.vertex_properties["weight_initialized"] = bools
        self.debug_msg("Enabled tracking of weight initialization")

    # randomly create node weights
    def add_node_weights(self, min=0.0, max=0.1, distribution=[1, 0, 0], set_initialized=False):
        self.debug_msg("Adding random weights between {} and {} to nodes.".format(min, max))
        num_nodes = float(self.graph.num_vertices())
        weights = self.graph.new_vertex_property("double")
        num_active = int(math.ceil(num_nodes * distribution[2]))
        num_inactive = int(math.ceil(num_nodes * distribution[1]))
        num_lurker = int(math.ceil(num_nodes * distribution[0]))
        weights_list = [uniform(0.0001, 1) for x in xrange(num_active)]
        weights_list.extend([uniform(min, max) for x in xrange(num_lurker)])
        weights_list.extend([uniform(0, 0) for x in xrange(num_inactive)])
        shuffle(weights_list)
        for ndx, n in enumerate(self.graph.vertices()):
            weights[n] = weights_list[ndx]
            if set_initialized:
                try:
                    self.graph.vertex_properties["weight_initialized"][n] = True
                except:
                    pass
        self.graph.vertex_properties["activity"] = weights


    # randomly create edge weights
    def add_edge_weights(self, min=0.0, max=0.1):
        self.debug_msg("Adding random weights between {} and {} to edges.".format(min, max))
        weights = self.graph.new_edge_property("double")
        for e in self.graph.edges():
            weights[e] = uniform(min, max)
        self.graph.edge_properties["activity"] = weights


    def prepare_eigenvalues(self):
        self.top_eigenvalues = np.asarray(self.graph.graph_properties['top_eigenvalues'])
        self.k1 = max(self.top_eigenvalues)


    # calculate eigenvalues, with a maximum of 100
    def calc_eigenvalues(self, num_ev=100):
        num_ev = min(100, num_ev)
        self.debug_msg("Extracting adjacency matrix!")
        A = adjacency(self.graph, weight=None)
        self.debug_msg("Starting calculation of {} Eigenvalues".format(num_ev))
        evals_large_sparse, evecs_large_sparse = largest_eigsh(A, num_ev * 2, which='LM')
        self.debug_msg("Finished calculating Eigenvalues")
        evs = sorted([float(x) for x in evals_large_sparse], reverse=True)[:num_ev]
        self.graph.graph_properties["top_eigenvalues"] = self.graph.new_graph_property("object", evs)


    # plot eigenvalue distribution
    def plot_eigenvalues(self):
        plt.figure(figsize=(8, 2))
        plt.scatter(real(self.top_eigenvalues), imag(self.top_eigenvalues), c=abs(self.top_eigenvalues))
        plt.xlabel(r"$Re(\kappa)$")
        plt.ylabel(r"$Im(\kappa)$")
        plt.tight_layout()
        plt.savefig(config.plot_dir + "eigenvalues/" + self.graph_name + "_adjacency_spectrum.pdf")
        plt.close("all")


    # colorful debug output :-)
    def debug_msg(self, msg):
        print "  \x1b[32m-GEN-\x1b[00m [\x1b[36m{}\x1b[00m] {}\x1b[00m".format(
            datetime.datetime.now().strftime("%H:%M:%S"), msg)
