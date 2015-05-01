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
        self.edge_list = None
        self.step_debug = "-"
        self.rand_iter_debug = "-"

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

    def activity_dynamics(self, store_weights=False, store_taus=False, empirical=False, scenario=None):
        # Collect required input
        activity_weight = np.asarray(self.get_node_weights("activity"))
        # Calculate deltax
        ratio_ones = (self.ratio * np.asarray(self.ones_ratio))
        intrinsic_decay = self.activity_decay(activity_weight, ratio_ones)
        extrinsic_influence = self.peer_influence(activity_weight)
        activity_delta = (intrinsic_decay + extrinsic_influence)*self.deltatau

        t = 1.0
        # Check if already converged/diverged
        if self.cur_iteration % self.store_iterations == 0:
            t = np.dot(activity_delta, activity_delta)
        # Debug output & convergence/divergence criteria check
        if t < self.converge_at and not empirical:
            self.debug_msg(" \x1b[32m>>>\x1b[00m Simulation for \x1b[32m'{}'\x1b[00m with \x1b[34mratio={}\x1b[00m and "
                            "\x1b[34mdtau={}\x1b[00m \x1b[34mconverged\x1b[00m at \x1b[34m{}\x1b[00m with "
                            "\x1b[34m{}\x1b[00m".format(self.graph_name, self.ratio, self.deltatau, self.cur_iteration+1,
                                                        t), level=1)
            self.converged = True
        if (t == float("Inf") or t == float("NaN")) and not empirical:
            self.debug_msg(" \x1b[31m>>>\x1b[00m Simulation for \x1b[32m'{}'\x1b[00m with \x1b[34mratio={}\x1b[00m and "
                            "\x1b[34mdtau={}\x1b[00m \x1b[31mdiverged\x1b[00m at \x1b[34m{}\x1b[00m with "
                            "\x1b[34m{}\x1b[00m".format(self.graph_name, self.ratio, self.deltatau, self.cur_iteration+1,
                                                        t), level=1)
            self.diverged = True
        # Set new weights
        self.update_node_weights("activity", activity_delta)

        # Check scenario stuff
        if scenario is not None and "Trolls" in scenario:
            self.check_and_set_trolls()
        elif scenario is not None and "Entities" in scenario:
            self.set_entities()

        # Store weights to file
        if ((store_weights and self.cur_iteration % self.store_iterations == 0) and not empirical) or ((self.converged or self.diverged)
                                                                                   and not empirical):
            self.weights_file.write(("\t").join(["%.8f" % x for x in self.get_node_weights("activity")]) + "\n")
        elif ((store_weights and self.cur_iteration % self.store_iterations == 0) and empirical) or ((self.converged or self.diverged)
                                                                                   and empirical):
            #self.debug_msg(" --> Sum of weights at \x1b[32m{}\x1b[30m is \x1b[32m{}\x1b[30m".format(self.cur_iteration, str(sum(activity_weight + activity_delta) * self.a_c * self.graph.num_vertices())), level=1)
            self.weights_file.write(str(sum(activity_weight + activity_delta) * self.a_c * self.graph.num_vertices()) + "\n")

        # Store taus to file
        if store_taus and empirical and self.cur_iteration % self.store_iterations == 0:
            tau = (float(self.tau_iter)+1)*self.deltatau/self.deltapsi
            tau += self.ratio_index
            self.taus_file.write(str(tau) + "\n")
        self.tau_iter += 1

        # Increment current iteration counter
        self.cur_iteration += 1

    def debug_msg(self, msg, level=0):
        if self.debug_level <= level:
            print "  \x1b[31m-SNWK-\x1b[00m [\x1b[36m{}\x1b[00m][\x1b[32m{}\x1b[00m][{}] \x1b[33m{}\x1b[00m".format(
                datetime.datetime.now().strftime("%H:%M:%S"), self.step_debug, self.rand_iter_debug, msg)

    def update_debug_info(self, step, rand_iter):
        self.step_debug = str(step)
        self.rand_iter_debug = str(rand_iter)

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

    def remove_users_by_num(self, num_users):
        bool_map = self.graph.new_vertex_property("bool")
        self.debug_msg(" --> Going to remove " + str(num_users) + " users of currently " +
                       str(self.graph.num_vertices()) + "...", level=1)
        user_sample = random.sample(range(0, self.graph.num_vertices()), num_users)
        for v in self.graph.vertices():
            if v in user_sample:
                bool_map[v] = 0
            else:
                bool_map[v] = 1
        self.graph.set_vertex_filter(bool_map)
        self.graph.purge_vertices()
        self.debug_msg(" --> Removed Users. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
                       str(self.graph.num_edges()) + " edges.", level=1)

    def remove_connections_by_percentage(self, percentage):
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
        self.debug_msg(" --> Removed Edges. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
                       str(self.graph.num_edges()) + " edges.", level=1)

    def remove_connections_by_num(self, num_connections):
        bool_map = self.graph.new_edge_property("bool")
        self.debug_msg(" --> Going to remove " + str(num_connections) + " connections of currently " +
                       str(self.graph.num_edges()) + "...", level=1)
        edge_sample = random.sample(range(0, self.graph.num_edges()), num_connections)
        for e in self.graph.edges():
            if self.graph.edge_index[e] in edge_sample:
                bool_map[e] = 0
            else:
                bool_map[e] = 1
        self.graph.set_edge_filter(bool_map)
        self.graph.purge_edges()
        self.debug_msg(" --> Removed Edges. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
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

    def add_connections_by_num(self, num_edges):
        if self.edge_list is None:
            self.edge_list = []
            self.debug_msg(" --> Starting calculation of edge list...", level=1)
            adj_matrix = self.A.todense()
            check_value = 0.0
            num_of_edges = (((self.graph.num_vertices() - 1) * self.graph.num_vertices()) / 2) - self.graph.num_edges()
            for i in range(0, self.graph.num_vertices()):
                if i % int((self.graph.num_vertices() / 10)) == 0 and i != 0:
                    self.debug_msg("  --> Covered " + str(int((float(i)/self.graph.num_vertices()) * 100)) +
                                   "% of vertices.", level=1)
                for j in range(0, self.graph.num_vertices()):
                    if adj_matrix[i, j] == check_value and i < j:
                        self.edge_list.append((i, j))
            self.debug_msg("  --> Covered 100% of vertices.", level=1)
            if len(self.edge_list) != num_of_edges:
                self.debug_msg("Error: The number of calculated edges did not meet the actual number of edges!", level=1)
                sys.exit("Program exit")
            else:
                self.debug_msg("Done. Number of calculated edges: " + str(len(self.edge_list)), level=1)
        random.shuffle(self.edge_list)
        old_num_edges = self.graph.num_edges()
        self.graph.add_edge_list(self.edge_list[0:num_edges])
        self.debug_msg(" --> Added " + str(num_edges) + " connections to the network (was: " + str(old_num_edges) +
                       ", now: " + str(self.graph.num_edges()) + ").", level=1)

    def add_trolls_by_num(self, num_trolls, negative_activity):
        #average_degree = int(np.mean(self.graph.vertex_properties["degree"].a))
        self.troll_ids = random.sample(range(0, self.graph.num_vertices()), num_trolls)
        for v_id in self.troll_ids:
            self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] = negative_activity

        #self.troll_ids = self.graph.add_vertex(num_trolls)
        #for troll in self.troll_ids:
        #    self.graph.vertex_properties["activity"][troll] = negative_activity
        #    targets = random.sample(range(0, self.graph.num_vertices()), average_degree)
        #    for tgt in targets:
        #        self.graph.add_edge(troll, tgt)
        self.debug_msg(" --> Added " + str(num_trolls) + " trolls with activity " + str(negative_activity) +
                       " to the network.", level=1)

    def add_entities_by_num(self, num_entities, activity):
        #average_degree = int(np.mean(self.graph.vertex_properties["degree"].a))
        self.entities_ids = random.sample(range(0, self.graph.num_vertices()), num_entities)
        for v_id in self.entities_ids:
            self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] = activity

        #for entity in self.entities_ids:
        #    self.graph.vertex_properties["activity"][entity] = activity
        #    targets = random.sample(range(0, self.graph.num_vertices()), average_degree)
        #    for tgt in targets:
        #        self.graph.add_edge(entity, tgt)
        self.debug_msg(" --> Added " + str(num_entities) + " entities with activity " + str(activity) +
                       " to the network.", level=1)

    def check_and_set_trolls(self, threshold=0):
        for v_id in self.troll_ids:
            if self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] > threshold:
                self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] = threshold

    def set_entities(self, activity=0.01):
        for v_id in self.entities_ids:
            self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] = activity