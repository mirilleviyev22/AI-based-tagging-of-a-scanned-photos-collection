import json
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def conf_calc(graph_obj, conf_mat, q, tol, alpha, central_idx, iteration, method='median'):
    #The function operations:
    #Compute and update the confidence vectors simultaneously with matrix multiplication: new_conf_mat = similarity_mat X conf_mat
    #Converting the resulting conf matrix into 1d array, computes the qth quantile in this array
    #The quantile value is set to be the threshold. Any cells lower than this threshold will be set to zero
    #Returns the updated confidence matrix

    new_conf_mat = np.zeros_like(conf_mat)
    similarity_mat = nx.to_scipy_sparse_array(graph_obj)
    node_list = list(graph_obj.nodes())
    id_mapping = {node_id: idx for idx, node_id in enumerate(node_list)}

    normalized_conf = conf_mat.copy()

    if iteration == 0 or method == 'average':
        # normalize each tag to balance between many (frequent) tags, to more rare tags
        # alpha = 1 -> scale_fac = 1 / tag frequency -> will give an advantage to unique \ rare tags that only appear in single images
        # alpha = 0 -> scale_fac = 1 -> no normalization, an advantage to the frequent tags
        tag_freq = np.sum(conf_mat > 0, axis=0)
        scale_fac = 1.0 / (1.0 + alpha * (np.log(tag_freq + 0.01)))

        neighbors_count = np.zeros(len(node_list))
        for node in node_list:
            idx = id_mapping[node]
            neighbors = graph_obj[node]
            count = 0
            for n_id, edge_data in neighbors.items():
                n_idx = id_mapping[n_id]
                if edge_data['weight'] > 0 and np.any(normalized_conf[n_idx] > 0):
                    count += 1
            neighbors_count[idx] = count

        print("\n[DEBUGGER] neighbor contributors:")
        print(f"{node_list[0]}: contribs of {neighbors_count[0]} connections at iteration {i}")
        print(f"{node_list[1]}: contribs of {neighbors_count[1]} connections at iteration {i}")

        for idx in range(conf_mat.shape[1]):
                if tag_freq[idx] > 0:
                    normalized_conf[:, idx] *= scale_fac[idx]

    if iteration > 0:
            if central_idx is not None and len(central_idx) > 0:
                for idx in central_idx:
                    # print(f"[DEBUGGER] Resetting central node {idx}: {normalized_conf[idx, :]}")
                    normalized_conf[idx, :] = 0.0

    if method == 'average' or iteration == 0:
        new_conf_mat = similarity_mat.dot(normalized_conf)

        denom = 1.0 + alpha/1.5 * np.log(neighbors_count + 0.01)
        denom[denom == 0] = 1.0
        new_conf_mat = new_conf_mat / denom[:, np.newaxis]

        print("[DEBUGGER] max conf after both normalizations:")
        print(f"Image 6: {np.max(new_conf_mat[0])}")
        print(f"Image 36: {np.max(new_conf_mat[1])}")

    if method == 'median' and iteration > 0:
        for node in node_list:
            current_node_idx = id_mapping[node]
            if central_idx is not None and current_node_idx in central_idx:
                continue

            neighbors_dict = graph_obj[node]

            if len(neighbors_dict) == 0:
                continue

            neighbors_sorted = sorted(neighbors_dict.items(), key=lambda item: item[1]['weight'], reverse=True)
            th = 0.3
            k_neighbors = [neighbor for neighbor in neighbors_sorted if neighbor[1]['weight'] >= th]
            if len(k_neighbors) < 3:
                k_neighbors = neighbors_sorted[:3]

            k_neighbors_id = [neighbors_id for neighbors_id, _ in k_neighbors]
            neighbor_idx = [id_mapping[neighbor_id] for neighbor_id in k_neighbors_id]
            neighbors_conf = upd_conf[neighbor_idx, :]

            #Initialize median confidence vector
            median_conf = np.zeros(upd_conf.shape[1])

            #Compute median for each tag using only non zero values
            for tag_idx in range(upd_conf.shape[1]):
                neighbor_values = neighbors_conf[:, tag_idx]
                non_zero_values = neighbor_values[neighbor_values > 0]

                #print(f"[DEBUGGER] Node {node}, tag {tag_idx}: neighbor_values = {neighbor_values}")
                #print(f"[DEBUGGER] Node {node}, tag {tag_idx}: non_zero_values = {non_zero_values}")

                if len(non_zero_values) > 0:
                    median_conf[tag_idx] = np.median(non_zero_values)
                    #print(f"[DEBUGGER] Node {node}, tag {tag_idx}: median = {median_conf[tag_idx]}")
                else:
                    median_conf[tag_idx] = 0.0
                    #print(f"[DEBUGGER] Node {node}, tag {tag_idx}: no non-zero values, set to 0")

            #print(f"[DEBUGGER] Node {node}: final median_conf = {median_conf}")
            #print(f"[DEBUGGER] Node {node}: median_conf unique values = {np.unique(median_conf)}")

            new_conf_mat[current_node_idx, :] = median_conf

    if central_idx is not None and len(central_idx) > 0:
        for idx in central_idx:
            # print(f"[DEBUGGER] Resetting central node {idx}: {normalized_conf[idx, :]}")
            new_conf_mat[idx, :] = 0.0

    all_values = new_conf_mat.ravel()
    non_zero_values = all_values[all_values > 0]
    print(f"[DEBUGGER] non_zero_values length: {len(non_zero_values)}")

    if len(non_zero_values) == 0:
        print("[DEBUGGER] WARNING: No positive values for quantile calculation!")
        return new_conf_mat, 0.0

    decay_boost = 0.7*np.exp(-2*i)
    for node in node_list:
        if node not in id_mapping:
            continue

        node_id = id_mapping[node]
        for n_id, edge_data in graph_obj[node].items():
            if edge_data['weight'] > 0.9 and n_id in id_mapping:
                nb_id = id_mapping[n_id]
                if np.any(normalized_conf[nb_id] > 0):
                    boost = decay_boost*edge_data['weight']*normalized_conf[nb_id]
                    new_conf_mat[node_id] += boost

    threshold = np.zeros(new_conf_mat.shape[0])
    for j in range (new_conf_mat.shape[0]):
        row = new_conf_mat[j]
        nonzero = row[row > 0]

        if len(nonzero) >= 2:
            th = np.quantile(nonzero, q)
            row[row < th] = 0
            new_conf_mat[j] = row
            threshold[j] = th

        elif len(nonzero) == 1:
            if nonzero[0] >= 1:
                threshold[j] = nonzero[0]
            else:
                new_conf_mat[j] = 0
                threshold[j] = 1


    #threshold = np.quantile(non_zero_values, q)
    #new_conf_mat[new_conf_mat < (threshold - tol)] = 0

    return new_conf_mat, threshold


def verify_propagation (conf_mat, results_csv, tags_data, tag_to_idx):
    pass

#def propagate_tags(graph_obj, results_csv, tags_json):

with open("graph_20250712_203514.json", "r") as f:
    graph_data = json.load(f)
graph_obj = nx.node_link_graph(graph_data, edges='links')

#graph_obj = nx.read_graphml("test_data/test_graph.graphml")
node_list = list(n.lower() for n in graph_obj.nodes())

#Convert weights to float if needed
for u, v, attr in graph_obj.edges(data=True):
    if 'weight' in attr and isinstance(attr['weight'], str):
        attr['weight'] = float(attr['weight'])

    #with open(tags_json, 'r') as file:
with open("test_tags.json", 'r') as file:
    tags_data_raw = json.load(file)

tags_data = {k.strip().lower(): v for k, v in tags_data_raw.items()}

name_votes = defaultdict(list)
all_tags = set()
central_id = set()
#Extract all unique tags from the JSON
for image_id, image_tags in tags_data.items():
    central_id.add(image_id)
    image_tags['tags'] = []

    if 'face_tags' in image_tags:
        for face_tag in image_tags['face_tags']:
            tag = face_tag['name'].lower()
            person_id = face_tag['face_id']
            name_votes[tag].append(person_id)
            image_tags['tags'].append(tag)
            all_tags.add(tag)

    if 'general_tags' in image_tags:
        for gen_tag in image_tags['general_tags']:
            tag = gen_tag.lower()
            image_tags['tags'].append(tag)
            all_tags.add(tag)

# Next step:
# Creating a mapping between name tag -> person id as appears in central images
# the person id is determined by voting, to minimize errors in verification further more

name_tag_to_id = {}
for tag, person_ids in name_votes.items():
    if len(person_ids) == 0:
        continue
    vote_counts = Counter(person_ids)
    winner = vote_counts.most_common(1)[0][0]
    name_tag_to_id['tag'] = winner

sorted_tags = sorted(all_tags)

print("=== DEBUG: Checking mappings ===")

print("\n1. All unique tags:")
print(sorted_tags)

print("\n2. Image tags per image:")
for image_id, image_tags in tags_data.items():
    print(f"{image_id}: {image_tags['tags']}")

print("\n3. Name tag to person ID mapping:")
print(name_tag_to_id)

print("\n4. Vote counts (for debugging):")
for tag_name, person_ids in name_votes.items():
    vote_counts = Counter(person_ids)
    print(f"{tag_name}: {dict(vote_counts)}")

#Create a mapping from tag to index in confidence vector
tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
print('tag to idx mapping:')
print(tag_to_idx)

##### NOTICE! The number of total images should be taken from the graph file, not the results.csv. it needs to be corrected later
    #results_pd = pd.read_csv(results_csv)
#results_pd = pd.read_csv("test_data/test_results.csv")
prop_tags = {img_path: [] for img_path in node_list}

id_mapping = {img_id: idx for idx, img_id in enumerate(node_list)}
print('[DEBUGGER] id_mapping:')
print(id_mapping)

central_indices = [id_mapping[image_id] for image_id in tags_data.keys()]
print("Central image indices:", central_indices)

#parameters for the confidence vectors:
total_images = len(graph_obj.nodes())#rows
total_tags = len(tag_to_idx) #cols

#Initializing the conf mat:
conf_mat = np.zeros((total_images, total_tags))

for image_id, image_tags in tags_data.items():
        for tag in image_tags['tags']:
            conf_mat[id_mapping[image_id], tag_to_idx[tag]] = 1.0

print("Confidence Vectors: ")
print(conf_mat)

n = 3
upd_conf = conf_mat.copy()

for i in range (n):
    upd_conf, th = conf_calc(graph_obj, upd_conf, q=0.65, tol=0.0, alpha=0.25, central_idx=central_indices,iteration=i, method='median')

    np.set_printoptions(precision=3, suppress=True)
    print('[DEBUGGER] Conf matrix at the end of iter %d:' % i)
    print(upd_conf)
    #print('[DEBUGGER] threshold = %.3f' % th, 'at the end of iter: %d' %i)
    for idx, t in enumerate(th):
        if t > 0:
            print(f"[DEBUGGER] image {node_list[idx]}: threshold={t:.3f}")

    #upd_conf = verify_propagation(upd_conf, results_csv, tags_data, tag_to_idx)

for image_id, image_tags in tags_data.items():
        for tag in image_tags['tags']:
            upd_conf[id_mapping[image_id], tag_to_idx[tag]] = 1.0

conf_th = 0
row_indices, col_indices = np.where(upd_conf > conf_th)

idx_to_node = {idx: node for idx, node in enumerate(node_list)}
#reverse mapping - idx to tag
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

for i in range(len(row_indices)):
    row_idx = row_indices[i]
    col_idx = col_indices[i]
    confidence = upd_conf[row_idx, col_idx]

    curr_img = idx_to_node[row_idx]
    curr_tag = idx_to_tag[col_idx]

    prop_tags[curr_img].append(
        {'tag': curr_tag,
         'confidence': round(float(confidence),4)
         })

print('tags propagated: ')
print(prop_tags)

with open("propagated_tags.json", 'w') as json_file:
    json.dump(prop_tags, json_file, indent=4)



