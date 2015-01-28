__author__ = 'Philipp Koncar'

from lib.util import *
from lib.generator import *
from multiprocessing import Pool

# The graph name used in the calculations
network_name = "Preferential_Attachment Network "

structural_changes = [
    "Add Edge ",
    "Remove Edge ",
    #"Remove Edge Spanning Tree ",
    "Add Vertex ",
    "Remove Vertex "
]


def prepare_plot_info(structural_change):
    network_name_full = ""
    subtitle = ""
    strategies = []
    strategy_labels = []
    if structural_change == "Add Vertex ":
        network_name_full = network_name + structural_change
        subtitle = "Iterations: 10 | Vertices added per iteration: 100 | Degree of new vertices: 5"
        strategies = ["random", "high_ev", "low_ev", "high_degree", "low_degree", "high_pr", "low_pr"]
        strategy_labels = ["Random", "High EV Centrality", "Low EV Centrality", "High Degree", "Low Degree", "High Pagerank", "Low Pagerank"]
    elif structural_change == "Remove Vertex ":
        network_name_full = network_name + structural_change
        subtitle = "Iterations: 10 | Vertices removed per iteration: 200"
        strategies = ["random", "high_ev", "low_ev", "high_degree", "low_degree", "high_pr", "low_pr"]
        strategy_labels = ["Random", "High EV Centrality", "Low EV Centrality", "High Degree", "Low Degree", "High Pagerank", "Low Pagerank"]
    elif structural_change == "Add Edge " or structural_change == "Remove Edge ":
        network_name_full = network_name + structural_change
        subtitle = "Iterations: 10"
        strategies = ["random", "high_to_high_ev", "high_to_low_ev", "low_to_low_ev", "high_to_high_degree", "high_to_low_degree", "low_to_low_degree", "high_to_high_pr", "high_to_low_pr", "low_to_low_pr"]
        strategy_labels = ["Random", "High To High EV Centrality", "High To Low EV Centrality", "Low To Low EV Centrality", "High To High Degree", "High To Low Degree", "Low To Low Degree", "High To High Pagerank", "High To Low Pagerank", "Low To Low Pagerank"]
    elif structural_change == "Remove Edge Spanning Tree ":
        network_name_full = network_name + structural_change.replace("Spanning Tree ", "")
        subtitle = "Iterations: 10 | Kept Spanning Tree"
        strategies = ["random_st", "high_to_high_ev_st", "high_to_low_ev_st", "low_to_low_ev_st", "high_to_high_degree_st", "high_to_low_degree_st", "low_to_low_degree_st", "high_to_high_pr_st", "high_to_low_pr_st", "low_to_low_pr_st"]
        strategy_labels = ["Random", "High To High EV Centrality", "High To Low EV Centrality", "Low To Low EV Centrality", "High To High Degree", "High To Low Degree", "Low To Low Degree", "High To High Pagerank", "High To Low Pagerank", "Low To Low Pagerank"]
    return network_name_full, subtitle, strategies, strategy_labels

if __name__ == '__main__':
    for str_change in structural_changes:
        network_name_full, subtitle, strategies, strategy_labels = prepare_plot_info(str_change)
        highest_evs = []
        average_evs = []
        median_evs = []
        edges_per_iter = []
        percent_per_iter = []
        for strategy in strategies:
            print "plotting: " + network_name_full + strategy
            highest_evs_, average_evs_, median_evs_, edges_per_iter_, percent_per_iter_ = load_and_prepare_eigenvalue_summary(network_name_full + strategy.replace("_", " ").title())
            highest_evs.append(highest_evs_)
            average_evs.append(average_evs_)
            median_evs.append(median_evs_)
            edges_per_iter.append(edges_per_iter_)
            percent_per_iter.append(percent_per_iter_)

        if "Spanning Tree" in str_change:
            suffix = " (Spanning Tree)"
        else:
            suffix = ""

        plot_eigenvalue_comparison(highest_evs, edges_per_iter[0], percent_per_iter[0], strategy_labels, network_name_full + "Strategy Comparison: Highest Eigenvalue", subtitle, ylabel="highest eigenvalue", filesuffix=suffix)
        #plot_eigenvalue_comparison(average_evs, edges_per_iter[0], percent_per_iter[0], strategy_labels, network_name_full + "Strategy Comparison: Average Eigenvalue", subtitle, ylabel="average eigenvalue", filesuffix=suffix)
        #plot_eigenvalue_comparison(median_evs, edges_per_iter[0], percent_per_iter[0], strategy_labels, network_name_full + "Strategy Comparison: Median Eigenvalue", subtitle, ylabel="median eigenvalue", filesuffix=suffix)

    print "DONE!"