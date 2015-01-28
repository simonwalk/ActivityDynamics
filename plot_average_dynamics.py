from lib.network import *
from lib.generator import *
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

b_vals = np.arange(0.0, 1.0, 0.001)
graph_names = ["Karate3"]
fx = "linear"
num_iterations = 10000

def main():
    for graph_name in graph_names:
        nw = Network(num_iterations, False, graph_name, 0.25, 0.25, fx=fx, debug=False, plot_with_labels=True)
        nw.debug_msg("Loading {}".format(config.graph_binary_dir+nw.fx_selector+"/"+graph_name+".gml"))
        nw.load_graph(config.graph_binary_dir+nw.fx_selector+"/"+graph_name+".gml")
        nw.debug_msg("Loading of {} successful!".format(config.graph_binary_dir+nw.fx_selector+"/"+graph_name+".gml"))

        clustering = nx.clustering(nw.graph, weight=None)
        pagerank = nx.pagerank(nw.graph, weight=None)
        hits = nx.hits(nw.graph)
        degree = nx.degree(nw.graph)

        f = open(config.graph_source_dir+"weights/"+fx + "/" + graph_name + "_last_iterations_var_b_weights.txt", "rb")

        avrg_node_weights = []
        for line in f:
            sl = line.strip().split("\t")
            s = sum([float(x) for x in sl])/len(sl)
            avrg_node_weights.append(s)

        f.close()

        plt.figure()

        plt.scatter(clustering.values(), avrg_node_weights, color="gray", s=degree.values()*30, cmap = plt.get_cmap('Spectral'))
        plt.xlabel("Clustering Coefficient")
        plt.ylabel("Average Node Weight")
        plt.gray()
        plt.grid(color="gray")
        plt.title("Weight vs. Clustering (size=degree)")
        labels = xrange(0,34,1)
        for label, x, y in zip(labels, clustering.values(), avrg_node_weights):
            plt.annotate(label,
                         xy = (x, y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
            )
        plt.savefig("plots/"+nw.graph_name + "_clustering_coefficient_scatterplot.pdf")
        plt.close("all")

        plt.figure()
        plt.scatter(pagerank.values(), avrg_node_weights, color="gray", s=degree.values()*30, cmap = plt.get_cmap('Spectral'))
        plt.xlabel("PageRank")
        plt.ylabel("Average Node Weight")
        plt.gray()
        plt.grid(color="gray")
        plt.title("Weight vs. PageRank (size=degree)")
        labels = xrange(0,34,1)
        for label, x, y in zip(labels, pagerank.values(), avrg_node_weights):
            plt.annotate(label,
                         xy = (x, y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
            )
        plt.savefig("plots/"+nw.graph_name + "_pagerank_scatterplot.pdf")
        plt.close("all")

        hubs = hits[0]
        auth = hits[1]

        plt.figure()
        plt.scatter(hubs.values(), avrg_node_weights, color="gray", s=degree.values()*30, cmap = plt.get_cmap('Spectral'))
        plt.xlabel("Hubs")
        plt.ylabel("Average Node Weight")
        plt.gray()
        plt.grid(color="gray")
        plt.title("Weight vs. Hubs (size=degree)")
        labels = xrange(0,34,1)
        for label, x, y in zip(labels, hubs.values(), avrg_node_weights):
            plt.annotate(label,
                         xy = (x, y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
            )
        plt.savefig("plots/"+nw.graph_name + "_hubs_scatterplot.pdf")
        plt.close("all")

        plt.figure()
        plt.scatter(auth.values(), avrg_node_weights, color="gray", s=degree.values()*30, cmap = plt.get_cmap('Spectral'))
        plt.xlabel("Authorities")
        plt.ylabel("Average Node Weight")
        plt.gray()
        plt.grid(color="gray")
        plt.title("Weight vs. Authorities (size=degree)")
        labels = xrange(0,34,1)
        for label, x, y in zip(labels, auth.values(), avrg_node_weights):
            plt.annotate(label,
                         xy = (x, y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
            )
        plt.savefig("plots/"+nw.graph_name + "_authorities_scatterplot.pdf")
        plt.close("all")

        plt.figure()
        plt.scatter(degree.values(), avrg_node_weights, color="gray", s=degree.values()*30, cmap = plt.get_cmap('Spectral'))
        plt.xlabel("Degree")
        plt.ylabel("Average Node Weight")
        plt.grid(color="gray")
        plt.gray()
        plt.grid(color="gray")
        plt.title("Weight vs. Degree (size=degree)")
        labels = xrange(0,34,1)
        for label, x, y in zip(labels, degree.values(), avrg_node_weights):
            plt.annotate(label,
                         xy = (x, y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
            )
        plt.savefig("plots/"+nw.graph_name + "_degrees_scatterplot.pdf")
        plt.close("all")

        plt.figure()
        #labels = ["Node {}".format(x) for x in xrange(0,34,1)]
        f = open(config.graph_source_dir+"weights/"+fx + "/" + graph_name + "_last_iterations_var_b_weights.txt", "rb")
        for idx, l in enumerate(f):
            plt.plot(b_vals, l.split("\t"), label="Node {}".format(idx))
        plt.xlabel("Value of b")
        plt.ylabel("Node Weights")
        plt.grid(color="gray")
        plt.title("Node weights at increasing b and a=0.02")
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6})
        plt.savefig("test.png")
        plt.close("all")


if __name__ == '__main__':main()