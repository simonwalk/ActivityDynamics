import pandas as pd
import numpy as np
import os
import shutil
from graph_tool.all import *

debug = False

#wiki_selector = 10
#wiki_selector = -2
wiki_selector = 10
is_wiki = False

instances = ["BEACHAPEDIA", "APBR", "CHARACTERDB", "SMWORG", "W15M", "AARDNOOT", "AUTOCOLLECTIVE", "CWW", "NOBBZ",
             "StackOverflow", "EnglishStackExchange", "HistoryStackExchange", "MathStackExchange", "BeerStackExchange",
             "NematodesWIKI", "AWAYCITY", "CDB", "CCC"]
folders = ["beachapedia_org_collab_network.txt.sorted_results",
           "apbrwiki_com_change_network.txt.sorted_results",
           "characterdb_cjklib_org_change_network.txt.sorted_results",
           "semantic-mediawiki_org_change_network.txt.sorted_results",
           "wiki_15m_cc_change_network.txt.sorted_results",
           "aardnoot_nl_change_network.txt.sorted_results",
           "autonomecollective_org_change_network.txt.sorted_results",
           "cumbriawindwatch_co_uk_change_network.txt.sorted_results",
           "nobbz_de_collab_network.txt.sorted_results",
           "StackOverflow", "EnglishStackExchange",
           "HistoryStackExchange", "MathStackExchange", "BeerStackExchange",
           "nematodes_org_collab_network.txt.sorted_results",
           "awaycity_com_collab_network.txt.sorted_results",
           "characterdb_cjklib_org_collab_network.txt.sorted_results",
           "events_ccc_de_collab_network.txt.sorted_results"]
instance = instances[wiki_selector]

#root_path = "/Volumes/DataStorage/Programming/"
root_path = "/Users/simon/Desktop/ActivityDynamics/results/graph_sources/collaboration_networks/"
root_path = "/Users/simon/Desktop/"
#root_path = "/opt/datasets/stackexchange/"
#root_path = "/Users/simon/Desktop/"

root_path_ratios = root_path + "ActivityDynamics/results/graph_binaries/empirical_input/"
storage_path = root_path_ratios + instance + "_empirical_input.txt"
init_weights_path = root_path_ratios + instance + "_weights.txt"

if is_wiki:
    source_path = root_path + "ActivityDynamics/results/graph_sources/collaboration_networks/"+folders[wiki_selector]+"/"
else:
    source_path = root_path + folders[wiki_selector]+"/"

print "Processing: {}".format(source_path)
binaries_path = root_path + "ActivityDynamics/results/graph_binaries/GT/"
gnames = [instance + "_run_0.gt"]
fnames = [binaries_path + instance + "/"]
copy_file = source_path + "net.gt"

gname = gnames[0]
fname = fnames[0]
if not os.path.exists(fname):
    os.makedirs(fname)
shutil.copy(copy_file, fname + gname)

# loading pickled files
df_posts = pd.read_pickle(source_path + "user_df_posts.ser")
df_replies = pd.read_pickle(source_path + "user_df_replies.ser")

print df_posts

print "----"

print df_replies

print "----"

graph = load_graph(source_path + "net.gt")
graph.clear_filters()

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
df_posts = pd.read_pickle(source_path + "user_df_posts.ser")
df_replies = pd.read_pickle(source_path + "user_df_replies.ser")
df_posts.fillna(0)
df_replies.fillna(0)
merged = df_posts + df_replies
merged.to_csv(source_path + "merged.csv", sep=";")
for i in xrange(0, max_row):
    init_users = set()
    posts_current = np.nansum(df_posts.ix[i,:])+1
    if i == 0:
        for ndx, id in enumerate(id_to_vertex_dict.keys()):
            if ndx+1 % 1000 == 0:
                print "  -- processing node {}".format(ndx+1)
            try:
                val = df_posts.iloc[i][id]
            except:
                print id
                val = 0
            if val < 1:
                val = 0
            if val > 0:
                print val
                init_users.add(str(id_to_vertex_dict[id]))
                #print id
            try:
                val = np.array(df_replies.iloc[i][id])
            except:
                #print id
                print val
                val = 0
            if val < 1:
                val = 0
            if val > 0:
                init_users.add(str(id_to_vertex_dict[id]))
    posts_next = np.nansum(df_posts.ix[i+1,:])+1
    replies_current = np.nansum(df_replies.ix[i,:])+1
    replies_next = np.nansum(df_replies.ix[i+1,:])+1
    num_users = np.sum((merged.iloc[[i]]).count())+1
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