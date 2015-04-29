__author__ = "Philipp Koncar"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "p.koncar@student.tugraz.at"
__status__ = "Development"

from lib.generator import *
from lib.scenario_network import ScenarioNetwork

plot_only = True

deltatau = 0.001
store_itas = 10
tid = 30
plot_fmt = "pdf"
rand_itas = 5

data_sets = ["BeerStackExchange",           # 0
             "HistoryStackExchange"]        # 1

emp_data_set = data_sets[0]

scenarios = [
             #"Remove Users",
             #"Remove Connections",
             #"Add Users",
             #"Add Connections",
             "Add Trolls",
             #"Add Entities"
            ]

percentage_steps = [5, 10, 15, 20, 25]


def create_network():
    bg = Generator(emp_data_set)
    bg.debug_msg("Loading network!")
    bg.load_graph(emp_data_set+"_run_"+str(0))
    bg.clear_all_filters()
    bg.calc_eigenvalues(2)
    bg.add_node_weights()
    bg.collect_colors()
    remove_self_loops(bg.graph)
    remove_parallel_edges(bg.graph)
    bg.draw_graph(0)
    bg.calc_vertex_properties()
    bg.store_graph(0)


def calc_activity(scenario):
    debug_msg(" *** Starting activity dynamics *** ")
    nw = ScenarioNetwork(False, emp_data_set, run=0, deltatau=deltatau, store_iterations=store_itas, tau_in_days=tid)
    fpath = nw.get_binary_filename(emp_data_set)
    nw.debug_msg("Loading {}".format(fpath), level=0)
    nw.load_graph(fpath)
    nw.debug_msg("Loaded network: " + str(nw.graph.num_vertices()) + " vertices, "
                 + str(nw.graph.num_edges()) + " edges")

    nw.prepare_eigenvalues()
    nw.create_folders()
    nw.get_empirical_input(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "_empirical.txt")
    nw.init_empirical_activity()
    nw.calculate_ratios()
    nw.set_ratio(0)
    nw.open_weights_files()
    nw.open_taus_files()
    nw.write_summed_weights_to_file()
    nw.write_initial_tau_to_file()
    scenario_marker = (len(nw.ratios) / 3) * 2
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
            nw.activity_dynamics(store_weights=True, store_taus=True, empirical=True)
        if i is scenario_marker - 1:
            nw.graph_copy = Graph(nw.graph)
            scenario_init_iter = nw.cur_iteration
            debug_msg(" --> Ratio == " + str(scenario_marker) + ". Saved activity: " +
                      str(sum(nw.graph.vertex_properties["activity"].a)) +
                      ", cur_iteration: " + str(scenario_init_iter))
    nw.close_weights_files()
    nw.close_taus_files()
    nw.add_graph_properties()
    nw.store_graph(0)

    # Helper functions
    def remove_users(percentage):
        debug_msg(" --> Doing remove users stuff...")
        nw.remove_users_by_percentage(percentage)
        nw.update_ones_ratio()
        nw.update_adjacency()
        debug_msg(" --> Done with removing users.")

    def remove_connections(percentage):
        debug_msg(" --> Doing remove edges stuff...")
        nw.remove_edges_by_percentage(percentage)
        debug_msg(" --> Done with removing edges.")

    def add_users():
        debug_msg(" --> Doing add users stuff...")
        nw.add_users_by_num(100)
        nw.update_ones_ratio()
        nw.update_adjacency()
        debug_msg(" --> Done with adding users.")

    def add_connections():
        debug_msg(" --> Doing add connections stuff...")

        debug_msg(" --> Done with adding connections.")

    def add_trolls():
        debug_msg(" --> Doing add troll stuff...")
        nw.add_trolls_by_num(2, -0.01)
        nw.update_ones_ratio()
        nw.update_adjacency()
        debug_msg(" --> Done with adding trolls.")

    def add_entities():
        debug_msg(" --> Doing add entities stuff...")
        nw.add_entities_by_num(2, 0.1)
        nw.update_ones_ratio()
        nw.update_adjacency()
        debug_msg(" --> Done with adding entities.")

    scenario_dispatcher = {"Remove Users": remove_users,
                           "Remove Connections": remove_connections,
                           "Add Users": add_users,
                           "Add Connections": add_connections,
                           "Add Trolls": add_trolls,
                           "Add Entities": add_entities}

    debug_msg(" *** Starting activity dynamics with scenario: " + scenario + " *** ")
    for rand_iter in range(0, rand_itas):
        debug_msg(" --> Starting iteration " + str(rand_iter + 1))
        nw.run = rand_iter
        nw.set_ratio(0)
        nw.open_weights_files(scenario)
        nw.open_taus_files(scenario)
        nw.cur_iteration = scenario_init_iter
        nw.graph = Graph(nw.graph_copy)
        debug_msg(" --> Reset graph. Activity to " + str(sum(nw.graph.vertex_properties["activity"].a)) +
                  ", cur_iteration to " + str(scenario_init_iter))

        nw.write_summed_weights_to_file()

        if "Remove" in scenario:
            scenario_dispatcher[scenario](20)
        else:
            scenario_dispatcher[scenario]()

        for i in range(scenario_marker, len(nw.ratios)):
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
                nw.activity_dynamics(store_weights=True, store_taus=True, empirical=True)

        nw.close_weights_files()
        nw.close_taus_files()

    calc_random_itas_average(emp_data_set, scenario, rand_itas, store_itas, nw.ratios[0], deltatau, delFiles=False)
    debug_msg(" *** Done with activity dynamics *** ")


if __name__ == '__main__':
    if not plot_only:
        create_network()
    for scenario in scenarios:
        if not plot_only:
            calc_activity(scenario)
        plot_scenario_results(emp_data_set, scenario, plot_fmt, rand_itas)
