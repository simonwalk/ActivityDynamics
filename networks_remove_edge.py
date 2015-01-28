__author__ = 'Philipp Koncar'

from lib.util import *
from lib.generator import *
from multiprocessing import Pool

num_iterations = 10
num_random_inits = 10  # only needed if random strategy is applied

num_nodes = 5000  # for preferential attachment and random

networks_used = [
                    #"Karate",
                    "Facebook",
                    #"Preferential_Attachment",
                    #"Random"
                ]

applied_strategies = [
                      #"random",
                      #"random_st",
                      "high_to_high_ev",
                      #"high_to_high_ev_st",
                      #"high_to_low_ev",
                      #"high_to_low_ev_st",
                      #"low_to_low_ev",
                      #"low_to_low_ev_st",
                      #"high_to_high_degree",
                      #"high_to_high_degree_st",
                      "high_to_low_degree",
                      #"high_to_low_degree_st",
                      #"low_to_low_degree",
                      #"low_to_low_degree_st",
                      #"high_to_high_pr",
                      #"high_to_high_pr_st",
                      #"high_to_low_pr",
                      #"high_to_low_pr_st",
                      #"low_to_low_pr",
                      #"low_to_low_pr_st"
                      ]

def create_network(graph_name, network, strategy):
    debug_msg("Starting with network: " + network)
    evs_per_iter = [[0 for x in xrange(num_iterations+1)] for x in xrange(num_iterations+1)]
    edges_per_iter = []
    percent_per_iter = []

    nw = Network(num_iterations, False, graph_name, run=0, plot_with_labels=True, debug_level=0)

    if not os.path.isfile(config.graph_binary_dir+"GML/"+network+"/"+network+"_run_"+str(0)+".gml"):
        bg = Generator(network, num_nodes)

        networks = {"Karate": bg.create_karate_graph,
                    "Facebook": bg.load_facebook_graph,
                    "Preferential_Attachment": bg.create_preferential_attachment,
                    "Random": bg.create_random_graph}

        bg.debug_msg("Creating " + network + " Network!")

        try:
            networks[network]()
        except KeyError:
            debug_msg("The given network is not found!")
            sys.exit("Program exit")

        bg.add_edge_weights(0.0, 0.1)
        bg.add_node_weights(0.1, 0.1, distribution=[0.4, 0.3, 0.3])
        bg.calc_eigenvalues(max(int(bg.graph.num_vertices()/2)-1, 1))
        bg.calc_vertex_properties()
        bg.collect_colors()
        bg.draw_graph(0, draw_labels=True)
        bg.store_gml(0)

    nw.load_graph(config.graph_binary_dir+"GML/"+network+"/"+network+"_run_"+str(0)+".gml")
    nw.prepare_eigenvalues()
    #nw.plot_eigenvalues(0)
    #nw.plot_gx(-20, 20)
    #nw.plot_fx(-1, 2)
    nw.collect_colors()
    del nw.graph.vertex_properties["pos"]
    nw.draw_graph(0)
    evs_per_iter[0] = nw.ew
    edges_per_iter.append(nw.graph.num_edges())
    percent_per_iter.append(nw.current_percent_of_connectivity())
    if "st" in strategy:
        num_array = nw.get_num_edges_per_iterations_array(nw.minimum_edges_for_spanning_tree(), num_iterations)
    else:
        num_array = nw.get_num_edges_per_iterations_array(0, num_iterations)
    debug_msg("Starting with strategy: " + strategy)
    for i in range(1, num_iterations + 1):
        debug_msg("Starting with iteration: " + str(i))
        nw.remove_edge_from_graph(num_array[i-1], strategy)
        nw.update_adjacency_matrix()
        nw.calc_eigenvalues(max(int(nw.graph.num_vertices()/2)-1, 1))
        nw.update_vertex_properties()
        nw.collect_colors()
        nw.draw_graph(i)
        #nw.plot_eigenvalues(i)
        evs_per_iter[i] = nw.ew
        edges_per_iter.append(nw.graph.num_edges())
        percent_per_iter.append(nw.current_percent_of_connectivity())
    debug_msg("Percentage: " + str(nw.current_percent_of_connectivity()))
    debug_msg("Nodes: " + str(nw.graph.num_vertices()))
    debug_msg("Edges: " + str(nw.graph.num_edges()))
    store_eigenvalue_summary(evs_per_iter, edges_per_iter, percent_per_iter, graph_name)

if __name__ == '__main__':
    for network in networks_used:
        base_graph = network + " Network Remove Edge "
        for strategy in applied_strategies:
            current_graph = base_graph + strategy.replace("_", " ").title()
            if strategy == "random":
                for iteration in range(0, num_random_inits):
                    graph_name_iter = current_graph + "_" + str(iteration)
                    create_network(graph_name_iter, network, strategy)
                store_eigenvalue_summary_average(current_graph, num_random_inits, True)
            else:
                create_network(current_graph, network, strategy)
            highest_evs, average_evs, median_evs, edges_per_iter, percent_per_iter = load_and_prepare_eigenvalue_summary(current_graph)
            title = base_graph + "(" + strategy.replace("_", " ") + " approach)"
            subtitle = "Iterations: " + str(num_iterations)
            plot_eigenvalues_summary(highest_evs, average_evs, median_evs, edges_per_iter, percent_per_iter, current_graph, title, subtitle)

    debug_msg("Done")