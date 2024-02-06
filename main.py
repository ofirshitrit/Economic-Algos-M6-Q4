import networkx as nx


# Step 1: find placement that max the sum of valuations
def max_sum_valuations(valuations):
    G = nx.Graph()

    # create edge for every player & object
    for player in range(len(valuations)):
        for obj, value in enumerate(valuations[player]):
            G.add_edge(f"player {player}", f"object {obj}", weight=value)

    # find maximum-value matching -> placement that maximize the valuations
    return nx.max_weight_matching(G)


# Step 2: Determine the prices such that the placement will be envy-free


if __name__ == '__main__':
    valuations = [[25, 40, 35], [40, 60, 35], [20, 40, 25]]
    print("Placement: ", max_sum_valuations(valuations))
