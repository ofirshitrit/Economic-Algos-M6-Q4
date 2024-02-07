import networkx as nx
import numpy as np
import cvxpy


# Step 1: find allocation that max the sum of valuations
def max_sum_valuations(valuations):
    G = nx.Graph()

    # create edge for every player & object
    for player in range(len(valuations)):
        for object in range(len(valuations[player])):
            G.add_edge(f"person {player}", f"room {object}", weight=valuations[player][object])

    # find maximum-value matching -> allocation that maximizes the valuations
    return nx.max_weight_matching(G)


# Step 2: Determine the prices such that the placement will be envy-free
def find_rent_prices(valuations, rent):
    allocation = max_sum_valuations(valuations)
    num_rooms = len(valuations[0])
    num_people = len(valuations)

    # Define price variables for each room
    price_rooms = [cvxpy.Variable() for _ in range(num_rooms)]

    # Constraints list
    constraints = []

    # The sum of the prices should be equal to the rent
    constraints.append(sum(price_rooms) == rent)

    # Prices need to be >= 0
    constraints.extend([price >= 0 for price in price_rooms])

    # Envy-free constraints for each person
    for person, room in allocation:
        person_index = int(person.split()[1])
        room_index = int(room.split()[1])
        for other_room_index in range(num_rooms):
            if other_room_index != room_index:
                constraints.append(
                    valuations[person_index][room_index] - price_rooms[room_index] >= valuations[person_index][
                        other_room_index] - price_rooms[other_room_index])

    # Define the optimization problem
    prob = cvxpy.Problem(cvxpy.Minimize(0), constraints)

    # Solve the problem
    prob.solve()

    # Print the results
    if prob.status == 'optimal':
        print("Allocation:")
        for person, room in allocation:
            print(f"{person} gets {room}")
        print("Prices:")
        for i in range(num_rooms):
            print(f"Room {i} rent: {np.round(price_rooms[i].value)}")
    else:
        print("infeasible")


if __name__ == '__main__':
    # valuations = [[20, 30, 40], [40, 30, 20], [30, 30, 30]]
    # valuations = [[25, 40, 35], [40, 60, 35], [20, 40, 25]]
    # valuations = [[150,0], [140,10]]
    valuations = [[20,30,40], [40,30,20],[30,40,20]]
    rent = 90
    allocation = max_sum_valuations(valuations)
    print("Allocation: ", allocation)
    # prices = find_rent_prices(valuations,rent)

    find_rent_prices(valuations, rent)
