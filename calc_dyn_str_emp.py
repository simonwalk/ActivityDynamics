from lib.util import *
from lib.generator import *
from lib.dynamic_network import DynamicNetwork
import time
from multiprocessing import Pool


emp_data_set = "BeerStackExchange"
mode = "months"  # Possible: "months", "days"
plot_fmt = "pdf"
tid = 30
deltatau = 0.001
store_itas = 1

manual_start_date = None  # Set to None to set start date automatically. (Type: datetime.date)
manual_network_epochs = None  # Set to None to setBeer number of network epochs automatically. (Type: int)
sleep_after_set = 3  # Time to wait after data is set automatically in order to check values (in seconds)


def create_network():
    bg = Generator(emp_data_set)
    bg.load_graph(emp_data_set+"_run_"+str(0))
    bg.clear_all_filters()
    bg.calc_eigenvalues(2)
    bg.track_weight_initialization()
    bg.add_node_weights(0.0, 0.0)
    bg.collect_colors()
    remove_self_loops(bg.graph)
    remove_parallel_edges(bg.graph)
    bg.draw_graph(0)
    bg.calc_vertex_properties()
    bg.store_graph(0)


def calc_activity():
    debug_msg(" *** Starting activity dynamics *** ")
    nw = DynamicNetwork(False, emp_data_set, run=0, deltatau=deltatau, store_iterations=store_itas, tau_in_days=tid)
    fpath = nw.get_binary_filename(emp_data_set)
    nw.debug_msg("Loading " + fpath)
    nw.load_graph(fpath)
    nw.debug_msg("Loaded network: " + str(nw.graph.num_vertices()) + " vertices, " + str(nw.graph.num_edges()) + " edges")

    if manual_start_date is None and manual_network_epochs is None:
        start_date, network_epochs = nw.get_epochs_info(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "_empirical.txt")
        debug_msg("Automatically set start date to: " + str(start_date))
        debug_msg("Automatically set network epochs to: " + str(network_epochs))
    elif manual_start_date is not None and manual_network_epochs is not None:
        start_date = manual_start_date
        network_epochs = manual_network_epochs
        debug_msg("Manually set start date to: " + str(start_date))
        debug_msg("Manually set network epochs to: " + str(network_epochs))
    else:
        debug_msg("Cannot automatically set only one value! Aborting...")
        sys.exit()
    debug_msg("Waiting " + str(sleep_after_set) + " seconds...")
    sleep(sleep_after_set)

    for epoch in range(0, network_epochs - 1):
        nw.reduce_network_to_epoch(start_date, epoch, mode=mode)
        nw.update_num_vertices_edges()
        nw.calc_eigenvalues_for_epoch(1)
        # TODO: remove calc_new_users_num_edges_vertices() here and in dynamic_network
        #nw.calc_new_users_num_edges_vertices()
        #print nw.num_new_user_vertices, nw.num_new_user_edges

    nw.get_empirical_input(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "_empirical.txt", start_date)
    nw.reduce_network_to_epoch(start_date, 0, mode=mode)
    nw.init_empirical_activity()
    nw.sapm.append(round(sum(nw.graph.vertex_properties["activity"].a) * nw.a_c * nw.graph.num_vertices()))
    nw.calculate_ratios()
    nw.set_ratio(0)
    nw.create_folders()
    nw.open_weights_files()
    nw.open_taus_files()
    nw.write_summed_weights_to_file()
    nw.write_initial_tau_to_file()
    for i in range(0, len(nw.ratios)):
        debug_msg("Starting activity dynamics for epoch: " + str(i+1))
        nw.reduce_network_to_epoch(start_date, i + 1, mode=mode)
        nw.update_ones_ratio()
        nw.update_adjacency()
        nw.debug_msg(" --> Sum of weights: \x1b[33m{}\x1b[0m with \x1b[33m{}\x1b[0m nodes".format(str(sum(nw.graph.vp["activity"].a) * nw.a_c * nw.graph.num_vertices()), nw.graph.num_vertices()), level=1)
        nw.update_dynamic_model_params(i)
        if i > 0:
            nw.update_activity()
            nw.update_init_empirical_activity()
            #nw.init_empirical_activity_new_users()
        nw.debug_msg(" --> Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                         "with \x1b[32m ratio={}\x1b[00m, "
                         "\x1b[32mk1={}\x1b[00m, \x1b[32m ratio-k1={}\x1b[00m, "
                         "\x1b[32mdtau={}\x1b[00m and \x1b[32mdpsi={}\x1b[00m "
                         "for \x1b[32m{} iterations\x1b[00m".format(emp_data_set, nw.ratio, nw.k1, nw.ratio-nw.k1, nw.deltatau, nw.deltapsi,
                                                                    int(nw.deltapsi/nw.deltatau)), level=1)
        for j in xrange(int(nw.deltapsi/nw.deltatau)):
            nw.activity_dynamics(store_weights=True, store_taus=True, empirical=True)
        nw.debug_msg(" --> Sum of weights: \x1b[33m{}\x1b[0m with \x1b[33m{}\x1b[0m nodes".format(str(sum(nw.graph.vp["activity"].a) * nw.a_c * nw.graph.num_vertices()), nw.graph.num_vertices()), level=1)
        nw.sapm.append(round(sum(nw.graph.vertex_properties["activity"].a) * nw.a_c * nw.graph.num_vertices()))
    nw.debug_msg("Observed activity over epochs: {}".format(nw.apm), level=1)
    nw.debug_msg("Simulated activity over epochs: {}".format(nw.sapm), level=1)
    nw.debug_msg(" --> Difference: {}".format(np.subtract(nw.sapm, nw.apm)), level=1)
    nw.close_weights_files()
    nw.close_taus_files()
    nw.add_graph_properties()
    nw.store_graph(0)
    debug_msg(" *** Done with activity dynamics *** ")

if __name__ == '__main__':
    create_network()
    calc_activity()
    empirical_result_plot_for_epochs(emp_data_set, mode, plot_fmt)
    sys.exit()
