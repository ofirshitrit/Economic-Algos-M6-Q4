import warnings
import networkx as nx
import sys
import numpy as np
import cvxpy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Comment this line to see prints of the logger
logger.setLevel(logging.WARNING)


# Step 1: find allocation that maximizes the sum of valuations
def max_sum_valuations(valuations):
    """
    Find allocation that maximizes the sum of valuations.

    Examples:
    >>> valuations = [[20, 30, 40], [40, 30, 20], [30, 30, 30]]
    >>> max_sum_valuations(valuations)
    [('person 0', 'room 2'), ('person 1', 'room 0'), ('person 2', 'room 1')]

    >>> valuations = [[150,0], [140,10]]
    >>> max_sum_valuations(valuations)
    [('person 0', 'room 0'), ('person 1', 'room 1')]
    """
    G = nx.Graph()

    # create edge for every player & object
    for person in range(len(valuations)):
        for room in range(len(valuations[person])):
            G.add_edge(f"person {person}", f"room {room}", weight=valuations[person][room])

    # find maximum-value matching -> allocation that maximizes the valuations
    alloc = nx.max_weight_matching(G)
    logger.info("alloc: %s", alloc)

    # Fixing the allocation to always have "person" first and then "room"
    fixed_alloc = {(person, room) if 'person' in person else (room, person)
                   for person, room in alloc}

    # Check if every person gets a room
    fixed_alloc = assign_rooms_for_unallocated_people(fixed_alloc, valuations)

    logger.info("fixed_alloc: %s", fixed_alloc)

    return fixed_alloc


def assign_rooms_for_unallocated_people(fixed_alloc, valuations):
    """
        Helper function that get room for person that didn't get and sorted the allocation.
        nx.max_weight_matching(G) every time return in diffrent order and not allway everyone get a room.
        Therefor I make sure that every person gets  a room and the order will be increase.

        Example:
        In this example nx.max_weight_matching(G) return {('person 0', 'room 0'), ('person 1', 'room 1'), ('person 2', 'room 2')}
        so person 3 need to get room 3 and then order the allocation

        >>> valuations = [[36, 34, 30, 0], [31, 36, 33, 0], [34, 30, 36, 0], [32, 33, 35, 0]]
        >>> fixed_alloc = {('person 0', 'room 0'), ('person 1', 'room 1'), ('person 2', 'room 2')}
        >>> assign_rooms_for_unallocated_people(fixed_alloc,valuations)
        [('person 0', 'room 0'), ('person 1', 'room 1'), ('person 2', 'room 2'), ('person 3', 'room 3')]
    """
    num_people = len(valuations)
    allocated_people = {person for person, _ in fixed_alloc}

    for person in range(num_people):
        if f"person {person}" not in allocated_people:
            available_rooms = [f"room {room}" for room in range(len(valuations[0]))]
            for _, room in fixed_alloc:
                if room in available_rooms:
                    available_rooms.remove(room)
            if available_rooms:
                fixed_alloc.add((f"person {person}", available_rooms[0]))
                logger.info("person %d assigned to room %d", person, int(available_rooms[0].split()[1]))
            else:
                logger.warning("person %d couldn't be assigned a room", person)

    # Sort fixed_alloc based on person indices
    fixed_alloc = sorted(fixed_alloc, key=lambda x: int(x[0].split()[1]))
    return fixed_alloc


# Step 2: Determine the prices such that the placement will be envy-free
def find_rent_with_nonnegative_prices(valuations, rent):
    """
    Find the prices such that the allocation will be envy-free and every person pays price >= 0.

    Examples:
    >>> valuations = [[20, 30, 40], [40, 30, 20], [30, 30, 30]]
    >>> rent = 90
    >>> find_rent_with_nonnegative_prices(valuations, rent)
    Allocation: [('person 0', 'room 2'), ('person 1', 'room 0'), ('person 2', 'room 1')]
    Room 0 rent: 31.0
    Room 1 rent: 27.0
    Room 2 rent: 31.0

    >>> valuations = [[25, 40, 35], [40, 60, 35], [20, 40, 25]]
    >>> rent = 50
    >>> find_rent_with_nonnegative_prices(valuations, rent)
    Allocation: [('person 0', 'room 2'), ('person 1', 'room 1'), ('person 2', 'room 0')]
    Room 0 rent: 8.0
    Room 1 rent: 28.0
    Room 2 rent: 15.0

    >>> valuations = [[20, 30, 40], [40, 30, 20], [30, 40, 20]]
    >>> rent = 90
    >>> find_rent_with_nonnegative_prices(valuations, rent)
    Allocation: [('person 0', 'room 2'), ('person 1', 'room 0'), ('person 2', 'room 1')]
    Room 0 rent: 30.0
    Room 1 rent: 31.0
    Room 2 rent: 29.0

    >>> valuations = [[150,0], [140,10]]
    >>> rent = 100
    >>> find_rent_with_nonnegative_prices(valuations, rent)
    Can not find allocation with prices >= 0

    >>> valuations = [[36, 34, 30, 0], [31, 36, 33, 0], [34, 30, 36, 0], [32, 33, 35, 0]]
    >>> rent = 100
    >>> find_rent_with_nonnegative_prices(valuations, rent)
    Can not find allocation with prices >= 0
    """

    allocation = max_sum_valuations(valuations)
    num_rooms = len(valuations[0])
    num_people = len(valuations)

    price_rooms = [cvxpy.Variable() for _ in range(num_rooms)]

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

    prob = cvxpy.Problem(cvxpy.Minimize(0), constraints)
    # Explicitly specify the solver to use for not showing the red warnings
    prob.solve(solver=cvxpy.ECOS)

    # Print the results
    if prob.status == 'optimal':
        print("Allocation:", allocation)
        logger.info("Prices:")
        for i in range(num_rooms):
            print(f"Room {i} rent: {np.round(price_rooms[i].value)}")
    else:
        print("Can not find allocation with prices >= 0")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Your problem is being solved with the ECOS solver by default.",
                            category=FutureWarning)

    import doctest

    doctest.testmod()
