
from random import randint
from numpy.random.mtrand import poisson

__author__ = 'simon'
from graph_tool.all import *

def corr(a, b):
    if a == b:
        return 0.999
    else:
        return 0.001

def main():
    g = load_graph("../graph_binaries/Karate_lin__0_iterations.gml")
    nweights = g.vertex_properties["weight"]
    spins = community_structure(g, 10000, 2)
    graph_draw(g, output_size=(800,800), output="karate.png",
                        vertex_size = prop_to_size(nweights, mi=0, ma=25),
                        vertex_fill_color = spins)

if __name__ == '__main__':main()
