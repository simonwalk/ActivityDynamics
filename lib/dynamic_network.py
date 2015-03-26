__author__ = "Philipp Koncar"
__version__ = "0.0.1"
__email__ = "p.koncar@student.tugraz.at"
__status__ = "Development"

# Base class
from network import Network

# Stuff from graph_tool
from graph_tool.all import *

import math

import sys

import numpy as np

# Needed to update eigenvalues
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

# Needed to calculate epochs
import datetime
from dateutil.relativedelta import relativedelta


class DynamicNetwork(Network):

    def __init__(self, directed, graph_name, run=1, converge_at=1e-16, deltatau=0.01, runs = 1,
                 deltapsi=0.0001, debug_level=1, store_iterations=1, ratios=[], ratio_index = 0, tau_in_days=30,
                 num_nodes=None):

        Network.__init__(self, directed, graph_name, run, converge_at, deltatau, runs, deltapsi, debug_level,
                         store_iterations, ratios, ratio_index, tau_in_days, num_nodes)

        # Lists to keep track of certain values over epochs
        self.k1_over_epochs = []
        self.num_vertices_over_epochs = []
        self.num_edges_over_epochs = []
        self.max_posts_per_day_over_epochs = []
        self.g_over_epochs = []
        self.max_q_over_epochs = []
        self.mu_over_epochs = []
        self.deltapsi_over_epochs = []

    def reduce_network_to_epoch(self, start_date, timedelta, mode=None):
        if mode is None:
            self.debug_msg("No mode given for reduce_network_to_epoch! Aborting...", level=1)
            sys.exit()
        elif mode is "months":
            temp_epoch = start_date + relativedelta(months=timedelta)
            end_day = datetime.date(temp_epoch.year, temp_epoch.month, 1) - datetime.timedelta(days=1)
        elif mode is "days":
            end_day = start_date + datetime.timedelta(days=timedelta)
        self.debug_msg("Getting network epoch from " + str(start_date) + " to " + str(end_day), level=1)
        self.graph.clear_filters()
        bool_map = self.graph.new_vertex_property("bool")
        for v in self.graph.vertices():
            if self.graph.vertex_properties["firstActivity"][v] > end_day:
                bool_map[v] = 0
            else:
                bool_map[v] = 1
        self.graph.set_vertex_filter(bool_map)
        self.debug_msg("Reduced network to " + str(self.graph.num_vertices()) + " vertices with " +
                       str(self.graph.num_edges()) + " edges.", level=1)

    def calc_eigenvalues_for_epoch(self, num_ev=100):
        num_ev = min(100, num_ev)
        self.debug_msg("Extracting adjacency matrix!")
        A = adjacency(self.graph, weight=None)
        self.debug_msg("Starting calculation of {} Eigenvalues".format(num_ev))
        evals_large_sparse, evecs_large_sparse = largest_eigsh(A, num_ev * 2, which='LM')
        self.debug_msg("Finished calculating Eigenvalues")
        evs = sorted([float(x) for x in evals_large_sparse], reverse=True)[:num_ev]
        self.graph.graph_properties["top_eigenvalues"] = self.graph.new_graph_property("object", evs)
        self.top_eigenvalues = self.get_eigenvalues()
        self.k1_over_epochs.append(max(self.top_eigenvalues))
        self.k1 = max(self.top_eigenvalues)

    def update_num_vertices_edges(self):
        self.num_vertices_over_epochs.append(self.graph.num_vertices())
        self.num_edges_over_epochs.append(self.graph.num_edges())

    def update_adjacency(self):
        self.A = adjacency(self.graph, weight=None)

    def update_ones_ratio(self):
        self.ones_ratio = [1.0] * self.graph.num_vertices()

    def plot_epoch(self, epoch):
        pos = self.graph.vertex_properties["pos"]
        graph_draw(self.graph, pos=pos, output=config.graph_dir + self.graph_name + ".png", fmt="png")

    def set_deltapsi(self, index):
        self.deltapsi = self.deltapsi_over_epochs[index]

    def calc_gs(self):
        for ep in range(0, len(self.dx)):
            self.g_over_epochs.append(self.max_posts_per_day_over_epochs[ep] / (math.sqrt(self.a_cs[ep] ** 2 +
                                                                          self.max_posts_per_day_over_epochs[ep] ** 2)))

    def calc_max_qs(self):
        for ep in range(0, len(self.dx)):
            self.max_q_over_epochs.append((self.max_posts_per_day_over_epochs[ep] * self.tau_in_days *
                                          self.num_vertices_over_epochs[ep]) / (2 * self.num_edges_over_epochs[ep] *
                                                                                self.g_over_epochs[ep]))

    def calc_mus(self):
        for ep in range(0, len(self.dx)):
            self.mu_over_epochs.append(self.max_q_over_epochs[ep] / self.a_cs[ep])

    def update_activity(self):
        self.graph.vp["activity"].a *= self.a_cs[self.ratio_index - 1]
        self.graph.vp["activity"].a *= self.num_vertices_over_epochs[self.ratio_index - 1]
        self.graph.vp["activity"].a /= self.a_c
        self.graph.vp["activity"].a /= self.graph.num_vertices()

    def calc_max_posts_per_day(self):
        for ep in range(0, len(self.dx)):
            self.max_posts_per_day_over_epochs.append(max(self.posts_per_user_per_day[0 + ep:1 + ep]))

    # Overridden methods
    def get_empirical_input(self, path, start_tau=0, end_tau=None, ac_per_taus=None):
        self.dx = []
        self.apm = []
        self.posts = []
        self.replies = []
        self.num_users = []
        self.init_users = []
        self.posts_per_user_per_day = []
        self.a_cs = []
        f = open(path, "rb")
        for ldx, line in enumerate(f):
            if ldx < 1:
                continue
            el = line.strip().split("\t")
            try:
              self.dx.append(float(el[1]))
            except:
                break
            self.apm.append(float(el[2]))
            self.posts.append(float(el[3]))
            self.replies.append(float(el[4]))
            try:
                self.init_users.append(el[6].split(","))
            except:
                self.init_users.append(["dummy"])
            num_users = float(el[5]) + 1
            self.num_users.append(num_users)
            self.posts_per_user_per_day.append(float(el[3])/num_users/self.tau_in_days)
        f.close()
        self.calc_acs()
        self.calc_max_posts_per_day()
        self.max_posts_per_day = Network.calc_max_posts_per_day(self)#self.max_posts_per_day_over_epochs[-1]
        self.calc_gs()
        self.g_per_month = self.g_over_epochs[-1]
        self.calc_max_qs()
        self.max_q = self.max_q_over_epochs[-1]
        self.calc_mus()
        self.deltapsi_over_epochs = self.mu_over_epochs
        self.debug_msg("max_posts_per_day_over_epochs: {}".format(self.max_posts_per_day_over_epochs), level=1)
        self.debug_msg("max_q_over_epochs: {}".format(self.max_q_over_epochs), level=1)
        self.debug_msg("deltapsi_over_epochs: {}".format(self.deltapsi_over_epochs), level=1)
        self.debug_msg("max_posts_per_day: {}".format(self.max_posts_per_day), level=1)
        self.debug_msg("a_c_over_epochs: {}".format(self.a_cs), level=1)
        self.debug_msg("kappa_1_over_epochs: {}".format(self.k1_over_epochs), level=1)

    def calc_acs(self):
        for ep in range(0, len(self.dx)):
            self.a_cs.append((self.replies[ep] + self.posts[ep]) / self.num_vertices_over_epochs[ep])
        self.set_ac(0)

    def calculate_ratios(self):
        for ep in range(0, len(self.dx)):
            activity_current = self.apm[ep]
            activity_next = activity_current-self.dx[ep]
            self.ratio = self.k1_over_epochs[ep] - math.log(activity_next/activity_current) / self.deltapsi_over_epochs[ep]
            self.ratio -= 0.03 * activity_current / (self.a_cs[ep] * self.num_vertices_over_epochs[ep])
            self.ratios.append(self.ratio)
        self.debug_msg("ratios ({}): {}".format(len(self.ratios), self.ratios), level=1)

    def get_node_weights(self, name):
        return np.delete(np.array(self.graph.vp[name].a), np.arange(self.graph.num_vertices(), self.num_vertices))

    def update_node_weights(self, name, added_weight):
        z = np.zeros((self.num_vertices-self.graph.num_vertices(),), dtype=np.int)
        added_weight = np.concatenate((added_weight, z), axis=1)
        self.graph.vertex_properties[name].a += added_weight

    def add_graph_properties(self):
        Network.add_graph_properties(self)
        self.set_graph_property("object", self.k1_over_epochs, "k1_over_epochs")
        self.set_graph_property("object", self.num_vertices_over_epochs, "num_vertices_over_epochs")
        self.set_graph_property("object", self.num_edges_over_epochs, "num_edges_over_epochs")
        self.set_graph_property("object", self.g_over_epochs, "g_over_epochs")
        self.set_graph_property("object", self.max_q_over_epochs, "max_q_over_epochs")
        self.set_graph_property("object", self.mu_over_epochs, "mu_over_epochs")
        self.set_graph_property("object", self.deltapsi_over_epochs, "deltapsi_over_epochs")
        self.debug_msg("*** Successfully added epochs data to graph ***", level=1)

    def debug_msg(self, msg, level=0):
        if self.debug_level <= level:
            print "  \x1b[34m-DNWK-\x1b[00m [\x1b[36m{}\x1b[00m][\x1b[32m{}\x1b[00m] \x1b[33m{}\x1b[00m".format(
                datetime.datetime.now().strftime("%H:%M:%S"), self.run, msg)
