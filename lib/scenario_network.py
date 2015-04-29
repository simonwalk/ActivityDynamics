__author__ = "Philipp Koncar"
__version__ = "0.0.1"
__email__ = "p.koncar@student.tugraz.at"
__status__ = "Development"

from lib.util import *


class ScenarioNetwork(Network):

    def __init__(self, directed, graph_name, run=1, converge_at=1e-16, deltatau=0.01, runs=1,
                 deltapsi=0.0001, debug_level=1, store_iterations=1, ratios=[], ratio_index=0, tau_in_days=30,
                 num_nodes=None):

        Network.__init__(self, directed, graph_name, run, converge_at, deltatau, runs, deltapsi, debug_level,
                         store_iterations, ratios, ratio_index, tau_in_days, num_nodes)

    def open_weights_files(self, suffix=""):
        if suffix is not "":
            suffix = "_" + suffix
        folder = config.graph_source_dir + "weights/" + self.graph_name + "/"
        wname = self.graph_name + "_" + str(self.store_iterations) + "_" + \
                str(float(self.deltatau)).replace(".", "") + "_" + str(self.ratio).replace(".", "") + "_run_" + \
                str(self.run) + suffix + "_weights.txt"
        self.weights_file_path = folder+wname
        self.weights_file = open(self.weights_file_path, "wb")

    def open_taus_files(self, suffix=""):
        if suffix is not "":
            suffix = "_" + suffix
        folder = config.graph_source_dir + "weights/" + self.graph_name + "/"
        wname = self.graph_name + "_" + str(self.store_iterations) + "_" + \
                str(float(self.deltatau)).replace(".", "") + "_" + str(self.ratio).replace(".", "") + "_run_" + \
                str(self.run) + suffix + "_taus.txt"
        self.taus_file_path = folder+wname
        self.taus_file = open(self.taus_file_path, "wb")

    def debug_msg(self, msg, level=0):
        if self.debug_level <= level:
            print "  \x1b[31m-SNWK-\x1b[00m [\x1b[36m{}\x1b[00m][\x1b[32m{}\x1b[00m] \x1b[33m{}\x1b[00m".format(
                datetime.datetime.now().strftime("%H:%M:%S"), self.run, msg)

    def update_adjacency(self):
        self.A = adjacency(self.graph, weight=None)

    def update_ones_ratio(self):
        self.ones_ratio = [1.0] * self.graph.num_vertices()

    def remove_users_by_percentage(self, percentage):
        bool_map = self.graph.new_vertex_property("bool")
        num_affected_users = int((self.graph.num_vertices() / 100) * percentage)
        self.debug_msg(" --> Going to remove " + str(percentage) + "% of users (" + str(num_affected_users)
                       + " of currently " + str(self.graph.num_vertices()) + ")...", level=1)
        user_sample = random.sample(range(0, self.graph.num_vertices()), num_affected_users)
        for v in self.graph.vertices():
            if v in user_sample:
                bool_map[v] = 0
            else:
                bool_map[v] = 1
        self.graph.set_vertex_filter(bool_map)
        self.debug_msg(" --> Removed Users. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
                       str(self.graph.num_edges()) + " edges.", level=1)

    def remove_edges_by_percentage(self, percentage):
        bool_map = self.graph.new_edge_property("bool")
        num_affected_edges = int((self.graph.num_edges() / 100) * percentage)
        self.debug_msg(" --> Going to remove " + str(percentage) + "% of edges (" + str(num_affected_edges)
                       + " of currently " + str(self.graph.num_edges()) + ")...", level=1)
        edge_sample = random.sample(range(0, self.graph.num_edges()), num_affected_edges)
        for e in self.graph.edges():
            if self.graph.edge_index[e] in edge_sample:
                bool_map[e] = 0
            else:
                bool_map[e] = 1
        self.graph.set_edge_filter(bool_map)
        self.debug_msg(" --> Removed Users. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
                       str(self.graph.num_edges()) + " edges.", level=1)
