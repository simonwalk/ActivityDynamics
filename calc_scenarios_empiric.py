__author__ = "Philipp Koncar"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "p.koncar@student.tugraz.at"
__status__ = "Development"

import matplotlib
matplotlib.use('Agg')

from lib.generator import *
from lib.scenario_network import ScenarioNetwork

plot_only = False

deltatau = 0.001
store_itas = 10
tid = 30
plot_fmt = "pdf"
rand_itas = 10

data_sets = ["BeerStackExchange",           # 0
             "BitcoinStackExchange",        # 1
             "ElectronicsStackExchange",    # 2
             "PhysicsStackExchange",        # 3
             "GamingStackExchange",         # 4
             "AskUbuntu",                   # 5
             "ComplexOperations",           # 6
             "BioInformatics",              # 7
             "CSDMS",                       # 8
             "Neurolex",                    # 9
             "PracticalPlants",             #10
             "BlockLand",                   #11
             "DotaWiki"]                    #12

emp_data_set = data_sets[10]


experiments = ["Random",
               "Informed"
              ]


scenarios = [
             #"Remove Users",
             #"Remove Connections",
             "Add Users",

             #"Add Connections",
             #"Add Trolls",
             #"Add Entities"
            ]


step_values = {"Remove Users": [1, 5, 10],
               "Remove Connections": [10, 30, 50],
               "Add Users": [1, 5, 10],
               "Add Connections": [10, 30, 50],
               "Add Trolls": [1, 5, 10],
               "Add Entities": [1, 5, 10]}

legend_suffix = {"Remove Users": "% of Users",
                 "Remove Connections": "% of Collaborative Edges",
                 "Add Users": "% of Users",
                 "Add Connections": "% of Collaborative Edges",
                 "Add Trolls": "Trolls",
                 "Add Entities": "Incentivized Users"}

iter_setup = {"Random": rand_itas,
              "Informed": 1}


def create_network():
    bg = Generator(emp_data_set)
    bg.debug_msg("Loading network!")
    bg.load_graph(emp_data_set+"_run_"+str(0))
    bg.clear_all_filters()
    bg.calc_eigenvalues(2)
    bg.add_node_weights()
    #bg.collect_colors()
    remove_self_loops(bg.graph)
    remove_parallel_edges(bg.graph)
    #bg.draw_graph(0)
    bg.calc_vertex_properties()
    bg.store_graph(0)


def calc_activity(experiment, scenario):
    debug_msg(" *** Starting activity dynamics with : " + experiment + " *** ")
    nw = ScenarioNetwork(False, emp_data_set, run=0, deltatau=deltatau, store_iterations=store_itas, tau_in_days=tid,
                         ratios=[])
    fpath = nw.get_binary_filename(emp_data_set)
    nw.debug_msg("Loading {}".format(fpath), level=0)
    nw.load_graph(fpath)
    nw.debug_msg("Loaded network: " + str(nw.graph.num_vertices()) + " vertices, "
                 + str(nw.graph.num_edges()) + " edges", level=1)

    nw.prepare_eigenvalues()
    nw.create_folders()
    nw.calc_average_degree()
    nw.get_empirical_input(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "_empirical.txt")
    nw.init_empirical_activity()
    nw.calculate_ratios()
    nw.set_ratio(0)
    nw.open_weights_files()
    #nw.open_taus_files()
    nw.write_summed_weights_to_file()
    #nw.write_initial_tau_to_file()
    scenario_marker = int((len(nw.ratios) / float(3)) * 2)
    debug_msg(" --> Simulate scenario activity after: " + str(scenario_marker) + " ratios.")
    for i in xrange(len(nw.ratios)):
        debug_msg("Starting activity dynamics for ratio: " + str(i+1))
        nw.debug_msg(" --> Sum of weights: {}".format(sum(nw.get_node_weights("activity"))), level=1)
        nw.set_ac(i)
        nw.set_ratio(i)
        nw.reset_tau_iter()
        nw.debug_msg(" --> Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                         "with \x1b[32m ratio={}\x1b[00m and "
                         "\x1b[32mdtau={}\x1b[00m and \x1b[32mdpsi={}\x1b[00m "
                         "for \x1b[32m{} iterations\x1b[00m".format(emp_data_set, nw.ratio, nw.deltatau, nw.deltapsi,
                                                                    int(nw.deltapsi/nw.deltatau)),
                             level=1)
        for j in xrange(int(nw.deltapsi/nw.deltatau)):
            nw.activity_dynamics(store_weights=False, store_taus=False, empirical=True)
        nw.write_summed_weights_to_file()
        if i is scenario_marker - 1:
            nw.graph_copy = Graph(nw.graph)
            scenario_init_iter = nw.cur_iteration
            debug_msg(" --> Ratio == " + str(scenario_marker) + ". Saved activity: " +
                      str(sum(nw.graph.vertex_properties["activity"].a)) +
                      ", cur_iteration: " + str(scenario_init_iter))
    nw.close_weights_files()
    #nw.close_taus_files()
    nw.add_graph_properties()
    nw.store_graph(0)

    # Helper functions
    def remove_users(strategy, num):
        debug_msg(" --> Doing remove users stuff...")
        nw.remove_users_by_num(strategy, nw.get_num_users_by_percentage(num))
        nw.update_ones_ratio()
        nw.update_adjacency()
        debug_msg(" --> Done with removing users.")

    def remove_connections(strategy, num):
        debug_msg(" --> Doing remove edges stuff...")
        nw.remove_connections_by_num(strategy, nw.get_num_edges_by_percentage(num))
        nw.update_adjacency()
        debug_msg(" --> Done with removing edges.")

    def add_users(strategy, num):
        debug_msg(" --> Doing add users stuff...")
        nw.add_users_by_num(strategy, nw.get_num_users_by_percentage(num))
        nw.update_ones_ratio()
        nw.update_adjacency()
        debug_msg(" --> Done with adding users.")

    def add_connections(strategy, num):
        debug_msg(" --> Doing add connections stuff...")
        nw.add_connections_by_num(strategy, nw.get_num_edges_by_percentage(num))
        nw.update_adjacency()
        debug_msg(" --> Done with adding connections.")

    def add_trolls(strategy, num):
        debug_msg(" --> Doing add troll stuff...")
        negative_activity = (10 / nw.a_c / nw.graph.num_vertices())
        nw.add_trolls_by_num(strategy, num, -negative_activity)
        nw.update_ones_ratio()
        nw.update_adjacency()
        debug_msg(" --> Done with adding trolls.")

    def add_entities(strategy, num):
        debug_msg(" --> Doing add entities stuff...")
        added_activity = (10 / nw.a_c / nw.graph.num_vertices()) / int(nw.deltapsi/nw.deltatau)
        nw.add_entities_by_num(strategy, num, added_activity)
        #nw.update_ones_ratio()
        #nw.update_adjacency()
        debug_msg(" --> Done with adding entities.")

    scenario_dispatcher = {"Remove Users": remove_users,
                           "Remove Connections": remove_connections,
                           "Add Users": add_users,
                           "Add Connections": add_connections,
                           "Add Trolls": add_trolls,
                           "Add Entities": add_entities}

    debug_msg(" *** Starting activity dynamics with scenario: " + scenario + " *** ")
    for step, step_value in enumerate(step_values[scenario]):
        debug_msg(" --> Starting step " + str(step + 1) + " with value " + str(step_value))
        for rand_iter in range(0, iter_setup[experiment]):
            debug_msg(" --> Starting iteration " + str(rand_iter + 1))
            nw.update_debug_info(scenario, experiment[0], step + 1, rand_iter + 1)
            nw.run = rand_iter
            nw.set_ratio(0)
            nw.open_weights_files(experiment + "_" + scenario + "_" + str(step_value))
            #nw.open_taus_files(scenario + "_" + str(step_value))
            nw.cur_iteration = scenario_init_iter
            nw.graph = Graph(nw.graph_copy)
            debug_msg(" --> Reset graph. Activity to " + str(sum(nw.graph.vertex_properties["activity"].a)) +
                      ", cur_iteration to " + str(scenario_init_iter))
            nw.write_summed_weights_to_file()
            old_num_vertices = nw.graph.num_vertices()
            old_num_edges = nw.graph.num_edges()
            scenario_dispatcher[scenario](experiment, step_value)
            nw.update_k1()
            nw.update_ratios(scenario_marker)
            nw.add_to_result_list(experiment + scenario + " " + str(step_value) + " " + str(rand_iter + 1) + ": " +
                                  str(round(nw.k1, 3)) + " " + str(nw.graph.num_vertices() - old_num_vertices) + " " +
                                  str(nw.graph.num_edges() - old_num_edges))
            for i in range(scenario_marker, len(nw.ratios)):
                debug_msg("Starting activity dynamics for ratio: " + str(i+1))
                nw.debug_msg(" --> Sum of weights: {}".format(sum(nw.get_node_weights("activity"))), level=1)
                nw.set_ac(i)
                nw.set_ratio(i)
                nw.reset_tau_iter()
                nw.debug_msg(" --> Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                                 "with \x1b[32m ratio={}\x1b[00m and "
                                 "\x1b[32mdtau={}\x1b[00m and \x1b[32mdpsi={}\x1b[00m "
                                 "for \x1b[32m{} iterations\x1b[00m".format(emp_data_set, nw.ratio, nw.deltatau,
                                                                            nw.deltapsi, int(nw.deltapsi/nw.deltatau)),
                             level=1)
                for j in xrange(int(nw.deltapsi/nw.deltatau)):
                    nw.activity_dynamics(store_weights=False, store_taus=False, empirical=True, scenario=scenario)
                nw.write_summed_weights_to_file()
            nw.close_weights_files()
            #nw.close_taus_files()
        if experiment is "Random":
            debug_msg(" --> Calculating random average...")
            calc_random_itas_average(emp_data_set, scenario, step_value, rand_itas, store_itas, nw.ratios[0], deltatau,
                                     delFiles=True)
    nw.write_results_list(emp_data_set, scenario, experiment)
    debug_msg(" *** Done with activity dynamics *** ")


if __name__ == '__main__':
    if not plot_only:
        create_network()
    for scenario in scenarios:
        for experiment in experiments:
            if not plot_only:
                calc_activity(experiment, scenario)
        plot_scenario_results(emp_data_set, scenario, step_values[scenario], plot_fmt, rand_itas,
                              legend_suffix[scenario])
