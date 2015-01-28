from tests.network_tests import *
from lib.generator import *
import datetime
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt


def print_hline(char='='):
    print str().center(80, char)


def print_header(*text):
    print_hline()
    print_l(text)
    print_hline()


def print_l(*text):
    print '[', datetime.datetime.now().replace(microsecond=0), ']\t',
    if isinstance(text, str):
        print text
    else:
        for i in text:
            print str(i),
        print ''


def plot_degree_dist(graph, filename):
    degree_dist = defaultdict(int)
    for v in graph.vertices():
        degree_dist[v.out_degree() + v.in_degree()] += 1
    plt.clf()
    x = range(min(degree_dist.keys()), max(degree_dist.keys()) + 1)
    plt.plot(x, [degree_dist[i] for i in x])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.savefig(filename if filename.endswith('.png') else filename + '.png')


def main():
    do_all = raw_input("Execute all tests?(y/n):") == 'y'

    if do_all or raw_input("Execute normal vs. degree corrected sbm?(y/n)") == 'y':
        nodes_per_block = 500
        num_blocks = 10
        bg = Generator('small_normal_sbm', num_nodes)
        self_block = 0.1
        other_block = 0.0001
        power_exp = 2.2
        bg.create_stochastic_blockmodel_graph(blocks=num_blocks, size=[nodes_per_block, nodes_per_block, nodes_per_block], self_block_connectivity=self_block, other_block_connectivity=other_block, power_exp=None)
        plot_degree_dist(bg.graph, 'small_normal_sbm_degdist')
        bg.draw_graph(0, draw_labels=False, size_property="degree")

        bg = Generator('small_deg_corr_sbm', num_nodes)
        bg.create_stochastic_blockmodel_graph(blocks=num_blocks, size=[nodes_per_block, nodes_per_block, nodes_per_block], self_block_connectivity=self_block, other_block_connectivity=other_block, power_exp=power_exp)
        plot_degree_dist(bg.graph, 'small_deg_corr_sbm_degdist')
        bg.draw_graph(0, draw_labels=False, size_property="degree")

    if do_all or raw_input("Execute fast tests?(y/n)") == 'y':
        nodes_per_block = 10
        num_blocks = 3
        for i in xrange(7):
            print_header("generate small sbm: " + str(num_blocks) + " blocks a " + str(nodes_per_block) + ' nodes')
            draw_sbm(name='small_sbm_relative' + str(num_blocks) + 'b_' + str(nodes_per_block) + 'n', self_block_connectivity=0.1, other_block_connectivity=0.01, size=[nodes_per_block] * num_blocks)
            nodes_per_block *= 2
        graph_name = 'small_sbm_absolute'
        bg = Generator(graph_name, num_nodes)
        bg.create_stochastic_blockmodel_graph(blocks=10, size=[10, 10, 100, 100, 200, 200, 200, 500, 500, 100], scale='absolute', self_block_connectivity=1500, other_block_connectivity=20)
        bg.draw_graph(0)

    if do_all or raw_input("Execute std sbm?(y/n):") == 'y':
        print_header("generate std sbm")
        draw_sbm(name='diff_sizes_test')

    if do_all or raw_input("Execute lined up sbm?(y/n):") == 'y':
        print_header('generate std lined up sbm')
        graph_name = 'lined_up_test_std'
        bg = Generator(graph_name, num_nodes)
        my_mat = bg.create_sbm_lined_up_matrix(blocks=10)
        bg.create_stochastic_blockmodel_graph(blocks=10, size=[100], connectivity_matrix=my_mat, scale='relative')
        bg.draw_graph(0)

        print_header('generate std lined up sbm')
        graph_name = 'lined_up_test_diff_con'
        bg = Generator(graph_name, num_nodes)
        my_mat = bg.create_sbm_lined_up_matrix(blocks=10, self_block_connectivity=[0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9, 1], other_block_connectivity=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        bg.debug_msg("Creating scaled BlockModel network!")
        bg.create_stochastic_blockmodel_graph(blocks=10, size=[100], connectivity_matrix=my_mat, scale='relative')
        bg.draw_graph(0)

    sizes = [10, 50, 100, 200, 500]
    blocks = 5
    if do_all or raw_input("Execute equal sizes?(y/n):") == 'y':
        for i in sizes:
            print_header('generate sbm equal sizes:', i)
            draw_sbm(name='equal_sizes_' + str(i), size=[i] * blocks)
    sizes = [5, 10, 20, 30]
    if do_all or raw_input("Execute multiply by blockidx sizes?(y/n):") == 'y':
        for i in sizes:
            print_header('generate sbm sizes = multiply by blockidx:', i)
            draw_sbm(name='mult_sizes_' + str(i), size=[i * (j + 1) for j in xrange(blocks)])

    if do_all or raw_input("Execute pow by blockidx sizes?(y/n):") == 'y':
        for i in sizes[:1]:
            print_header('generate sbm sizes = pow by blockidx:', i)
            draw_sbm(name='pow_sizes_' + str(i), size=[int(pow(i, (j + 1))) for j in xrange(blocks)])


if __name__ == '__main__':
    main()
