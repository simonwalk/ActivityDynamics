from __future__ import division

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

        self.graph_copy = None
        self.troll_ids = []
        self.entities_ids = []

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
        self.graph.purge_vertices()
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
        self.graph.purge_edges()
        self.debug_msg(" --> Removed Users. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
                       str(self.graph.num_edges()) + " edges.", level=1)

    def add_users_by_num(self, num_users):
        average_degree = int(np.mean(self.graph.vertex_properties["degree"].a))
        old_num_users = self.graph.num_vertices()
        self.debug_msg(" --> Going to add " + str(num_users) + " new users with degree " + str(average_degree) +
                       " to the network...", level=1)
        for i in range(0, num_users):
            targets = random.sample(range(0, self.graph.num_vertices()), average_degree)
            v = self.graph.add_vertex()
            for t in targets:
                self.graph.add_edge(v, t)
        self.debug_msg(" --> Added " + str(num_users) + " users to the network (was: " + str(old_num_users) + ", now: " +
                       str(self.graph.num_vertices()) + ").", level=1)

    def add_edges_by_num(self, num_edges):
        pass

    def add_trolls_by_num(self, num_trolls, negative_activity):
        average_degree = int(np.mean(self.graph.vertex_properties["degree"].a))
        self.troll_ids = self.graph.add_vertex(num_trolls)
        for troll in self.troll_ids:
            self.graph.vertex_properties["activity"][troll] = negative_activity
            targets = random.sample(range(0, self.graph.num_vertices()), average_degree)
            for tgt in targets:
                self.graph.add_edge(troll, tgt)
        self.debug_msg(" --> Added " + str(num_trolls) + " trolls with activity " + str(negative_activity) +
                       " to the network.", level=1)

    def add_entities_by_num(self, num_entities, activity):
        average_degree = int(np.mean(self.graph.vertex_properties["degree"].a))
        self.entities_ids = self.graph.add_vertex(num_entities)
        for entity in self.entities_ids:
            self.graph.vertex_properties["activity"][entity] = activity
            targets = random.sample(range(0, self.graph.num_vertices()), average_degree)
            for tgt in targets:
                self.graph.add_edge(entity, tgt)
        self.debug_msg(" --> Added " + str(num_entities) + " entities with activity " + str(activity) +
                       " to the network.", level=1)
