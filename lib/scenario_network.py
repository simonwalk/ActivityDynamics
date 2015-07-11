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
        self.graph_copy_2 = None
        self.scenario_ids = []
        # self.edge_list = None
        self.experiment_debug = "-"
        self.scenario_debug = "-"
        self.step_debug = "-"
        self.rand_iter_debug = "-"
        self.entities_activity = 0
        self.average_degree = 0
        self.results_list = []
        self.A_copy = None

    def open_weights_files(self, suffix=""):
        if suffix is not "":
            suffix = "_" + suffix
        folder = config.graph_source_dir + "weights/" + self.graph_name + "/"
        wname = self.graph_name + "_" + str(self.store_iterations) + "_" + \
                str(float(self.deltatau)).replace(".", "") + "_run_" + \
                str(self.run) + suffix + "_weights.txt"
        self.weights_file_path = folder+wname
        self.weights_file = open(self.weights_file_path, "wb")

    def open_weights_files_append(self, suffix=""):
        if suffix is not "":
            suffix = "_" + suffix
        folder = config.graph_source_dir + "weights/" + self.graph_name + "/"
        wname = self.graph_name + "_" + str(self.store_iterations) + "_" + \
                str(float(self.deltatau)).replace(".", "") + "_run_" + \
                str(self.run) + suffix + "_weights.txt"
        self.weights_file_path = folder+wname
        self.weights_file = open(self.weights_file_path, "a")

    def open_taus_files(self, suffix=""):
        if suffix is not "":
            suffix = "_" + suffix
        folder = config.graph_source_dir + "weights/" + self.graph_name + "/"
        wname = self.graph_name + "_" + str(self.store_iterations) + "_" + \
                str(float(self.deltatau)).replace(".", "") + "_run_" + \
                str(self.run) + suffix + "_taus.txt"
        self.taus_file_path = folder+wname
        self.taus_file = open(self.taus_file_path, "wb")

    def activity_dynamics(self, store_weights=False, store_taus=False, empirical=False, scenario=None):
        # Collect required input
        activity_weight = np.asarray(self.get_node_weights("activity"))
        # Calculate deltax
        ratio_ones = (self.ratio * np.asarray(self.ones_ratio))
        if scenario is not None and ("Trolls" in scenario):
            for v in self.scenario_ids:
                ratio_ones[v] = float(0)
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

        # Store weights to file
        if ((store_weights and self.cur_iteration % self.store_iterations == 0) and not empirical) or ((self.converged or self.diverged)
                                                                                   and not empirical):
            self.weights_file.write(("\t").join(["%.8f" % x for x in self.get_node_weights("activity")]) + "\n")
        elif ((store_weights and self.cur_iteration % self.store_iterations == 0) and empirical) or ((self.converged or self.diverged)
                                                                                   and empirical):
            #self.debug_msg(" --> Sum of weights at \x1b[32m{}\x1b[30m is \x1b[32m{}\x1b[30m".format(self.cur_iteration, str(sum(activity_weight + activity_delta) * self.a_c * self.graph.num_vertices())), level=1)
            self.weights_file.write(str(sum(activity_weight + activity_delta) * self.a_c * self.graph.num_vertices()) + "\n")

        if scenario is not None and "Entities" in scenario:
            self.set_entities()

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
            print "  \x1b[31m-SNWK-\x1b[00m [\x1b[36m{}\x1b[00m][\x1b[36m{}\x1b[00m][\x1b[36m{}\x1b[00m]" \
                  "[\x1b[33m{}\x1b[00m][\x1b[32m{}\x1b[00m][{}] \x1b[33m{}\x1b[00m".format(
                datetime.datetime.now().strftime("%H:%M:%S"), self.experiment_debug, self.scenario_debug,
                self.ratio_index + 1, self.step_debug, self.rand_iter_debug, msg)

    def update_debug_info(self, scenario, experiment, step, rand_iter):
        self.experiment_debug = experiment
        self.scenario_debug = scenario
        self.step_debug = str(step)
        self.rand_iter_debug = str(rand_iter)

    def update_adjacency(self):
        self.A = adjacency(self.graph, weight=None)

    def update_ones_ratio(self):
        self.ones_ratio = [1.0] * self.graph.num_vertices()

    def get_important_nodes_by_degree(self, num_users):
        sorted_list = sorted(enumerate(self.graph.vertex_properties["degree"].a), key=lambda x: x[1], reverse=True)
        matching_vertices = []
        for i in range(0, num_users):
            matching_vertices.append(sorted_list[i][0])
        return matching_vertices

    def get_important_edges_by_degree(self, num_edges):
        sorted_list = sorted(enumerate(self.graph.vertex_properties["degree"].a), key=lambda x: x[1], reverse=True)
        edge_list = []
        i = 0
        while len(set(edge_list)) < num_edges:
            for n in self.graph.vertex(sorted_list[i][0]).all_neighbours():
                if sorted_list[i][0] < n:
                    edge_list.append(self.graph.edge_index[self.graph.edge(sorted_list[i][0], n)])
                else:
                    edge_list.append(self.graph.edge_index[self.graph.edge(n, sorted_list[i][0])])
            i += 1
        edge_list = list(set(edge_list))
        return edge_list[0:num_edges]

    def get_num_users_by_percentage(self, percentage):
        num_affected_users = int(round((self.graph.num_vertices() / 100) * percentage))
        return num_affected_users

    def get_num_edges_by_percentage(self, percentage):
        num_affected_edges = int(round((self.graph.num_edges() / 100) * percentage))
        return num_affected_edges

    def get_num_missing_edges_by_percentage(self, percentage):
        num_of_edges = (((self.graph.num_vertices() - 1) * self.graph.num_vertices()) / 2) - self.graph.num_edges()
        num_affected_edges = int(round((num_of_edges / 100) * percentage))
        return num_affected_edges

    # def remove_users_by_percentage(self, strategy, percentage):
    #     bool_map = self.graph.new_vertex_property("bool")
    #     num_affected_users = int((self.graph.num_vertices() / 100) * percentage)
    #     self.debug_msg(" --> Going to remove " + str(percentage) + "% of users (" + str(num_affected_users)
    #                    + " of currently " + str(self.graph.num_vertices()) + ")...", level=1)
    #     user_sample = random.sample(range(0, self.graph.num_vertices()), num_affected_users)
    #     for v in self.graph.vertices():
    #         if v in user_sample:
    #             bool_map[v] = 0
    #         else:
    #             bool_map[v] = 1
    #     self.graph.set_vertex_filter(bool_map)
    #     self.graph.purge_vertices()
    #     self.debug_msg(" --> Removed Users. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
    #                    str(self.graph.num_edges()) + " edges.", level=1)

    def remove_users_by_num(self, strategy, num_users):
        bool_map = self.graph.new_vertex_property("bool")
        self.debug_msg(" --> Going to remove " + str(num_users) + " users of currently " +
                       str(self.graph.num_vertices()) + "...", level=1)
        if strategy is "Random":
            user_sample = random.sample(range(0, self.graph.num_vertices()), num_users)
        else:
            user_sample = self.get_important_nodes_by_degree(num_users)
        for v in self.graph.vertices():
            if v in user_sample:
                bool_map[v] = 0
            else:
                bool_map[v] = 1
        self.graph.set_vertex_filter(bool_map)
        self.graph.purge_vertices()
        self.debug_msg(" --> Removed Users. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
                       str(self.graph.num_edges()) + " edges.", level=1)

    # def remove_connections_by_percentage(self, strategy, percentage):
    #     bool_map = self.graph.new_edge_property("bool")
    #     num_affected_edges = int((self.graph.num_edges() / 100) * percentage)
    #     self.debug_msg(" --> Going to remove " + str(percentage) + "% of edges (" + str(num_affected_edges)
    #                    + " of currently " + str(self.graph.num_edges()) + ")...", level=1)
    #     edge_sample = random.sample(range(0, self.graph.num_edges()), num_affected_edges)
    #     for e in self.graph.edges():
    #         if self.graph.edge_index[e] in edge_sample:
    #             bool_map[e] = 0
    #         else:
    #             bool_map[e] = 1
    #     self.graph.set_edge_filter(bool_map)
    #     self.graph.purge_edges()
    #     self.debug_msg(" --> Removed Edges. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
    #                    str(self.graph.num_edges()) + " edges.", level=1)

    def remove_connections_by_num(self, strategy, num_connections):
        bool_map = self.graph.new_edge_property("bool")
        self.debug_msg(" --> Going to remove " + str(num_connections) + " connections of currently " +
                       str(self.graph.num_edges()) + "...", level=1)
        if strategy is "Random":
            edge_sample = random.sample(range(0, self.graph.num_edges()), num_connections)
        else:
            edge_sample = self.get_important_edges_by_degree(num_connections)
        for e in self.graph.edges():
            if self.graph.edge_index[e] in edge_sample:
                bool_map[e] = 0
            else:
                bool_map[e] = 1
        self.graph.set_edge_filter(bool_map)
        self.graph.purge_edges()
        self.debug_msg(" --> Removed Edges. Current Graph: " + str(self.graph.num_vertices()) + " vertices and " +
                       str(self.graph.num_edges()) + " edges.", level=1)

    def add_users_by_num(self, strategy, num_users):
        old_num_users = self.graph.num_vertices()
        self.debug_msg(" --> Going to add " + str(num_users) + " new users with degree " + str(self.average_degree) +
                       " to the network...", level=1)
        targets = self.get_important_nodes_by_degree(self.average_degree)
        for i in range(0, num_users):
            if strategy is "Random":
                targets = random.sample(range(0, self.graph.num_vertices()), self.average_degree)
            v = self.graph.add_vertex()
            for t in targets:
                self.graph.add_edge(v, t)
        self.debug_msg(" --> Added " + str(num_users) + " users to the network (was: " + str(old_num_users) + ", now: " +
                       str(self.graph.num_vertices()) + ").", level=1)

    def add_connections_by_num(self, strategy, num_edges):
        self.debug_msg(" --> Going to add " + str(num_edges) + " new edges to the network...", level=1)
        # if self.edge_list is None:
        #     self.edge_list = []
        #     self.debug_msg(" --> Starting calculation of edge list...", level=1)
        #     adj_matrix = self.A.todense()
        #     check_value = 0.0
        #     num_of_edges = (((self.graph.num_vertices() - 1) * self.graph.num_vertices()) / 2) - self.graph.num_edges()
        #     for i in range(0, self.graph.num_vertices()):
        #         if i % int((self.graph.num_vertices() / 10)) == 0 and i != 0:
        #             self.debug_msg("  --> Covered " + str(int((float(i)/self.graph.num_vertices()) * 100)) +
        #                            "% of vertices.", level=1)
        #         for j in range(0, self.graph.num_vertices()):
        #             if adj_matrix[i, j] == check_value and i < j:
        #                 self.edge_list.append((i, j))
        #     self.debug_msg("  --> Covered 100% of vertices.", level=1)
        #     if len(self.edge_list) != num_of_edges:
        #         self.debug_msg("Error: The number of calculated edges did not meet the actual number of edges!", level=1)
        #         sys.exit("Program exit")
        #     else:
        #         self.debug_msg("Done. Number of calculated edges: " + str(len(self.edge_list)), level=1)
        # random.shuffle(self.edge_list)
        if self.A_copy is None:
            self.A_copy = self.A.todense().copy()
        edge_list = set()
        if strategy is "Random":
            self.debug_msg(" --> Calculate random edge tuples...", level=1)
            a_copy = self.A_copy.copy()
            while True:
                edge_tuple = random.sample(range(0, self.graph.num_vertices()), 2)
                if a_copy[edge_tuple[0], edge_tuple[1]] == 0.0:
                    edge_list.add((edge_tuple[0], edge_tuple[1]))
                    a_copy[edge_tuple[0], edge_tuple[1]] = 1.0
                    a_copy[edge_tuple[1], edge_tuple[0]] = 1.0
                if len(edge_list) == num_edges:
                    break
        elif strategy is "Informed":
            self.debug_msg(" --> Calculate informed edge tuples...", level=1)
            high_list = self.get_important_nodes_by_degree(self.graph.num_vertices())
            a_copy = self.A_copy.copy()
            for i in high_list:
                for j in range(0, self.graph.num_vertices()):
                    if a_copy[i, j] == 0.0 and i != j:
                        edge_list.add((i, j))
                        a_copy[i, j] = 1.0
                        a_copy[j, i] = 1.0
                    if len(edge_list) == num_edges:
                        break
                if len(edge_list) == num_edges:
                    break
        edge_list = list(edge_list)
        old_num_edges = self.graph.num_edges()
        self.graph.add_edge_list(edge_list)
        self.debug_msg(" --> Added " + str(num_edges) + " connections to the network (was: " + str(old_num_edges) +
                       ", now: " + str(self.graph.num_edges()) + ").", level=1)

    def add_trolls_by_num(self, strategy, num_trolls, negative_activity):
        # if strategy is "Random":
        #     self.scenario_ids = random.sample(range(0, self.graph.num_vertices()), num_trolls)
        # else:
        #     self.scenario_ids = self.get_important_nodes_by_degree(num_trolls)
        # for v_id in self.scenario_ids:
        #     self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] = negative_activity

        #self.scenario_ids = self.graph.add_vertex(num_trolls)
        self.scenario_ids = []
        for i in range(0, num_trolls):
            v = self.graph.add_vertex()
            self.scenario_ids.append(v)
        for troll in self.scenario_ids:
            self.graph.vertex_properties["activity"][troll] = negative_activity
            if strategy is "Random":
                targets = random.sample(range(0, self.graph.num_vertices()), self.average_degree)
            else:
                targets = self.get_important_nodes_by_degree(self.average_degree)
            # targets = random.sample(range(0, self.graph.num_vertices()), average_degree)
            for tgt in targets:
                self.graph.add_edge(troll, tgt)
        self.debug_msg(" --> Added " + str(num_trolls) + " trolls with activity " + str(negative_activity) +
                       " to the network.", level=1)

    def add_entities_by_num(self, strategy, num_entities, activity):
        self.entities_activity = activity
        #average_degree = int(np.mean(self.graph.vertex_properties["degree"].a))
        if strategy is "Random":
            self.scenario_ids = random.sample(range(0, self.graph.num_vertices()), num_entities)
        else:
            self.scenario_ids = self.get_important_nodes_by_degree(num_entities)
        for v_id in self.scenario_ids:
            self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] += activity
        self.debug_msg(" --> Added " + str(num_entities) + " entities with " + str(activity) +
                       " added activity to the network.", level=1)

    def check_and_set_trolls(self, threshold=0):
        for v_id in self.scenario_ids:
            if self.graph.vertex_properties["activity"][v_id] > threshold:
                self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] = threshold

    def set_entities(self):
        for v_id in self.scenario_ids:
            self.graph.vertex_properties["activity"][self.graph.vertex(v_id)] += self.entities_activity

    def update_k1(self):
        evals_large_sparse, evecs_large_sparse = largest_eigsh(self.A, 2, which='LM')
        evs = sorted([float(x) for x in evals_large_sparse], reverse=True)[0]
        self.debug_msg("Old k1: " + str(self.k1), level=1)
        self.k1 = evs
        self.debug_msg("New k1: " + str(self.k1), level=1)

    def update_ratios(self, scenario_marker):
        self.debug_msg(" --> Updating ratios...")
        for i in range(scenario_marker, len(self.ratios)):
            activity_current = self.apm[i]
            activity_next = activity_current-self.dx[i]
            self.ratios[i] = self.k1 - math.log(activity_next/activity_current) / self.deltapsi
        self.debug_msg("ratios ({}): {}".format(len(self.ratios), self.ratios), level=1)

    def update_current_ratio(self):
        activity_current = self.apm[self.ratio_index]
        activity_next = activity_current-self.dx[self.ratio_index]
        self.ratio = self.k1 - math.log(activity_next/activity_current) / self.deltapsi

    def calc_average_degree(self):
        self.average_degree = max(1, int(round(np.mean(self.graph.vertex_properties["degree"].a))))
        self.debug_msg(" --> Calculated average degree: " + str(self.average_degree), level=1)

    def add_to_result_list(self, value):
        self.results_list.append(value)

    def write_results_list(self, graph_name, scenario, experiment):
        out_path = os.path.abspath(config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + "_" +
                                          scenario + "_" + experiment + "_structure_results.txt")
        myfile = open(out_path, "wb")
        for entry in self.results_list:
            myfile.write(entry + "\n")
        myfile.close()