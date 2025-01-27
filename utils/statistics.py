import torch
from collections import Counter
from dgl.data import CoraGraphDataset

def extract_by_bool(bool_list, value_list):
    result = []
    for i in range(len(bool_list)):
        if bool_list[i]:
            result.append(value_list[i])

    return result

def extract_by_index(index_list, value_list):
    result = []
    for index in index_list:
        result.append(value_list[index])

    return result

def count_labels(labels):
    label_counter = Counter(labels)
    return label_counter

def get_all_neighbors(graph):
    all_neighbors = []
    for node_index in range(graph.num_nodes()):
        neighbor_indices = graph.successors(node_index)
        all_neighbors.append(neighbor_indices.tolist())

    return all_neighbors

def get_all_2_hop_neighbors(graph, neighbors_list):
    all_2_hop_neighbors = []
    for neighbors in neighbors_list:
        labels = [neighbors]
        for neighbor in neighbors:
            labels.append(graph.successors(neighbor).tolist())
        all_2_hop_neighbors.append(labels)

    return all_2_hop_neighbors

def get_neighbors_labels(labels_list, neighbors_list):
    neighbors_labels = []
    for neighbors in neighbors_list:
        labels = []
        for neighbor in neighbors:
            labels.append(labels_list[neighbor])
        neighbors_labels.append(labels)

    return neighbors_labels

def wrong_node_analyse(index):
    test_node_degree = extract_by_bool(masks[2].tolist(), g.in_degrees().tolist())
    test_node_label = extract_by_bool(masks[2].tolist(), labels)
    test_node_neighbor = extract_by_bool(masks[2].tolist(), all_neighbors)
    test_node_2_hop_neighbor = extract_by_bool(masks[2].tolist(), all_2_hop_neighbors)

    wrong_node_degree = extract_by_index(index, test_node_degree)
    wrong_node_degree_counts = count_labels(wrong_node_degree)
    wrong_node_label = extract_by_index(index, test_node_label)
    wrong_node_label_counts = count_labels(wrong_node_label)
    wrong_node_neighbor = extract_by_index(index, test_node_neighbor)
    wrong_node_2_hop_neighbor = extract_by_index(index, test_node_2_hop_neighbor)

    print("Wrong Node Degree:", wrong_node_degree)
    print("Wrong Node Degree Counts:", wrong_node_degree_counts)
    print("Wrong Node Label:", wrong_node_label)
    print("Wrong Label Counts:", wrong_node_label_counts)
    print("Wrong Node Neighbor:", wrong_node_neighbor)
    print("Wrong Node Neighbor Label:", get_neighbors_labels(labels, wrong_node_neighbor))
    print("Wrong Node 2-Hop Neighbor:", wrong_node_2_hop_neighbor)


dataset = CoraGraphDataset()
g = dataset[0]
g = g.int()

all_neighbors = get_all_neighbors(g)
all_2_hop_neighbors = get_all_2_hop_neighbors(g, all_neighbors)

masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
labels = g.ndata["label"].tolist()

index_gcn = [0, 25, 27, 33, 56, 66, 84, 85, 91, 93, 94, 95, 126, 132, 136, 137, 143, 146, 161, 184, 185, 197, 200, 203, 204, 212, 216, 230, 249, 269, 273, 277, 281, 282, 290, 294, 295, 302, 305, 306, 307, 311, 312, 314, 315, 316, 320, 323, 324, 325, 327, 330, 331, 333, 336, 339, 350, 363, 370, 375, 381, 389, 392, 393, 394, 395, 397, 400, 403, 411, 416, 417, 418, 421, 430, 442, 443, 444, 445, 446, 458, 460, 462, 465, 467, 469, 470, 479, 480, 500, 503, 505, 523, 538, 547, 549, 555, 556, 560, 566, 568, 574, 583, 584, 585, 586, 594, 601, 602, 605, 606, 611, 615, 622, 631, 633, 637, 638, 640, 641, 645, 646, 647, 651, 652, 664, 665, 673, 677, 680, 689, 694, 697, 701, 705, 708, 715, 717, 718, 719, 728, 729, 730, 741, 745, 759, 760, 761, 764, 774, 776, 777, 781, 782, 788, 814, 834, 845, 846, 853, 854, 857, 858, 859, 861, 864, 865, 870, 872, 876, 878, 881, 882, 892, 894, 895, 898, 899, 903, 909, 912, 923, 931, 932, 945, 947, 956, 958, 961, 975, 982, 984, 991, 992, 994, 996, 997]
index_gcnii = [0, 27, 56, 66, 84, 85, 93, 94, 95, 126, 132, 137, 146, 161, 184, 197, 200, 201, 202, 203, 212, 215, 222, 273, 277, 282, 290, 295, 302, 305, 306, 307, 308, 311, 312, 315, 316, 320, 323, 327, 336, 339, 350, 361, 362, 363, 370, 381, 392, 393, 394, 395, 400, 403, 411, 416, 417, 418, 442, 445, 458, 460, 462, 465, 467, 469, 471, 479, 480, 500, 503, 520, 538, 547, 556, 560, 566, 574, 583, 585, 594, 597, 608, 615, 622, 624, 633, 637, 638, 640, 641, 644, 645, 646, 647, 648, 652, 664, 676, 677, 680, 686, 694, 696, 697, 701, 705, 717, 719, 725, 728, 729, 745, 748, 759, 760, 761, 762, 774, 776, 777, 824, 825, 854, 861, 870, 872, 876, 892, 912, 931, 945, 958, 982, 984, 991, 994, 997]

wrong_node_analyse(index_gcn)  # Wrong Node Degree Counts: Counter({2: 51, 1: 51, 3: 41, 4: 20, 6: 10, 5: 7, 8: 6, 7: 6, 10: 3, 9: 1, 11: 1})
wrong_node_analyse(index_gcnii)  # Wrong Node Degree Counts: Counter({2: 37, 1: 30, 3: 29, 4: 11, 6: 10, 5: 8, 8: 5, 7: 5, 10: 1, 9: 1, 16: 1})