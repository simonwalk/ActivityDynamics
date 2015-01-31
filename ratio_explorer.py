import pandas as pd
import numpy as np
import os
import shutil
from graph_tool.all import *

debug = False

wiki_selector = 10
#wiki_selector = -1

instances = ["BEACHAPEDIA", "APBR", "CHARACTERDB", "SMWORG", "W15M", "AARDNOOT", "AUTOCOLLECTIVE", "CWW", "NOBBZ",
             "StackOverflow", "EnglishStackExchange", "HistoryStackExchange", "MathStackExchange", "BeerStackExchange"]
folders = ["beachapedia_org_change_network.txt.sorted_results",
           "apbrwiki_com_change_network.txt.sorted_results",
           "characterdb_cjklib_org_change_network.txt.sorted_results",
           "semantic-mediawiki_org_change_network.txt.sorted_results",
           "wiki_15m_cc_change_network.txt.sorted_results",
           "aardnoot_nl_change_network.txt.sorted_results",
           "autonomecollective_org_change_network.txt.sorted_results",
           "cumbriawindwatch_co_uk_change_network.txt.sorted_results",
           "nobbz_de_change_network.txt.sorted_results",
           "StackOverflow", "EnglishStackExchange",
           "HistoryStackExchange", "MathStackExchange", "BeerStackExchange"]
instance = instances[wiki_selector]

root_path = "/Volumes/DataStorage/Programming/"
root_path = "/opt/datasets/stackexchange/"
#root_path = "/Users/simon/Desktop/"

root_path_ratios = root_path + "ActivityDynamics/results/graph_binaries/empirical_input/"
storage_path = root_path_ratios + instance + "_empirical_input.txt"
init_weights_path = root_path_ratios + instance + "_weights.txt"

source_path = root_path + folders[wiki_selector]+"/"

print "Processing: {}".format(source_path)
binaries_path = root_path + "ActivityDynamics/results/graph_binaries/GT/"
gnames = [instance + "_run_0.gt"]
fnames = [binaries_path + instance + "/"]
copy_file = source_path + "weighted_net.gt"

gname = gnames[0]
fname = fnames[0]
if not os.path.exists(fname):
    os.makedirs(fname)
shutil.copy(copy_file, fname + gname)

# loading pickled files
df_posts = pd.read_pickle(source_path + "user_df_posts.ser")
df_replies = pd.read_pickle(source_path + "user_df_replies.ser")

graph = load_graph(source_path + "weighted_net.gt")
graph.clear_filters()
#print "num nodes: {}".format(graph.num_vertices())
#print graph.vp["nodeID"].a
#print 47555 in graph.vp["nodeID"].a
#print df_posts[4]
#exit()

id_pmap = graph.vp["nodeID"]
id_to_vertex_dict = {int(id_pmap[v]): v for v in graph.vertices()}


# opening ratios file for activity dynamics simulation
f = open(storage_path, "wb")
g = open(init_weights_path, "wb")
max_row = len(df_posts.index)
f.write("dx\tagg_activity\tposts\treplies\tnum_users\tactive_user_ids\n")
print "Sum of Posts: {}".format(np.nansum(df_posts))
print "Sum of Replies: {}".format(np.nansum(df_replies))
print "Number of months: {}".format(len(df_posts))
print "Number of Users: {}".format(graph.num_vertices())


keys = set(id_to_vertex_dict.keys())
#print keys
df_posts = pd.read_pickle(source_path + "user_df_posts.ser")
df_replies = pd.read_pickle(source_path + "user_df_replies.ser")
assert set(df_posts.columns) == set(df_replies.columns)
print 'users in df:', len(df_posts.columns)
val = set(df_posts.columns) | set(df_replies.columns)
#print val
print 'overlap:', len(keys & val)
print keys - val
exit()

for i in xrange(0, max_row):
    init_users = set()
    posts_current = np.nansum(df_posts.ix[i,:])+1

    for id in id_to_vertex_dict.keys():
        try:
            val = df_posts.iloc[i][id]
        except:
            print id
            val = 0
        if val < 1:
            val = 0
        if val > 0:
            init_users.add(str(id_to_vertex_dict[id]))
        # try:
        #     val = np.array(df_replies.iloc[i][id])
        # except:
        #     print id
        #     val = 0
        # if val < 1:
        #     val = 0
        # if val > 0:
        #     init_users.add(str(id_to_vertex_dict[id]))
    posts_next = np.nansum(df_posts.ix[i+1,:])+1
    replies_current = np.nansum(df_replies.ix[i,:])+1
    replies_next = np.nansum(df_replies.ix[i+1,:])+1
    num_users = df_posts.ix[i+1,:].count()+1
    dx = (replies_current + posts_current) - (replies_next + posts_next)
    current_activity = (replies_current + posts_current)
    # write to file
    f.write(str(dx) + "\t" + str(current_activity) + "\t" + str(posts_current) + "\t" +
            str(replies_current) + "\t" + str(num_users) + "\t" + (",").join(init_users) + "\n")

    if debug:
        print "Month {}".format(i)
        print "------------------------"
        print "  Posts: {} - {}".format(posts_current, posts_next)
        print "  Replies: {} - {}".format(replies_current, replies_next)
        print "  Users: {}".format(num_users)
    if i == max_row-2:
        break

f.close()
g.close()