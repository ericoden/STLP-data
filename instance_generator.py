"""
Code for generating instances of the STL Problem
"""
import os
import numpy as np
import pandas as pd
import logging
import argparse
from numpy.random import default_rng
from collections import namedtuple
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

# Parameters
LENGTH = 2000 
WIDTH = LENGTH # Cluster centers are drawn from LENGTH x WIDTH rectangle
N_CLUSTERS = 5 # Number of clusters (i.e., modes of the GMM)
SHAPE_BOUND = 200 # Covariance matrix given by A*A^T, where A is populated by
                  # values from [-SHAPE_BOUND, SHAPE_BOUND]
MIN_OD_DIST = 300 # minimum distance between pickup and delivery
MIN_LOAD = 8 # minimum load size of a shipment
MAX_LOAD = 44 # maximum load size of a shipment


Node = namedtuple("Node", "x y q d")


def distance(a: Node, b: Node) -> float:
    """
    Computes the Euclidean distance between two nodes
    """
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def generate_instance(C: int, seed: int):
    """
    Generates an instance as described in the paper.
    """
    logging.info(f'Generating an instance with {C} shippers with seed {seed}')
    # final_seed is set to seed + 1000 * C, so that no two problem sizes have
    # the same seed, if seed is kept within 0-999.
    final_seed = seed + 1000 * C
    rng = default_rng(final_seed)
    # Cluster Generation
    mus = np.array([rng.uniform(low=0, high=LENGTH, size=2) 
                    for _ in range(N_CLUSTERS)]) 
    sigmas = [] 
    for _ in range(N_CLUSTERS):
        a = rng.uniform(low=-SHAPE_BOUND, high=SHAPE_BOUND, size=(2,2))
        sigma = np.matmul(a, a.transpose())
        sigmas.append(sigma)
    sigmas = np.array(sigmas)

    Node = namedtuple("Node", "x y q d")


    # Location Generation
    nodes = []
    for i in range(2 * C):
        if i < C:
            index = rng.choice(N_CLUSTERS)
            x, y = rng.multivariate_normal(mean=mus[index], cov=sigmas[index])
            q = rng.uniform(low=MIN_LOAD, high=MAX_LOAD)
            nodes.append(Node(x=x, y=y, q=q, d=0))
        else:
            pickup = nodes[i - C]
            flag = False
            while flag is False: # deliveries must be MIN_OD_DIST from pickups
                index = rng.choice(N_CLUSTERS)
                x,y = rng.multivariate_normal(mean=mus[index], 
                                              cov=sigmas[index])
                node = Node(x=x, y=y, q=0, d=0)
                if distance(node, pickup) > MIN_OD_DIST:
                    flag = True
                    nodes.append(node)

    # calculate average distance to K nearest neighbors (K=5)
    K = 5
    average_distances = []
    for i in range(2 * C):
        node = nodes[i]
        distances = np.array([distance(node, other_node) 
                            for other_node in nodes])
        idx = np.argpartition(distances, K)
        average_distance = sum(distances[idx[:K]]) / K
        average_distances.append(average_distance)
    average_distances = np.array(average_distances)

    high_cutoff = np.percentile(average_distances, q= 100 * (1 / 3))
    medium_cutoff = np.percentile(average_distances, q= 100 * (2 / 3))

    demand_levels = np.array([3 if d < high_cutoff else 
                              (2 if d < medium_cutoff else 1) 
                              for d in average_distances])

    for index in range(len(nodes)):
        nodes[index] = nodes[index]._replace(d=demand_levels[index])               
    return nodes

def plot_instance(nodes, save=False):
    """
    Plots an instance of the STL problem.
    """
    c_dict = {1: 'lightgrey', 2: 'darkgrey', 3: 'black'}
    label_dict = {1: "Low Demand", 2: "Medium Demand", 3: "High Demand"}
    fig, ax = plt.subplots()
    for d in [1, 2, 3]:
        ax.scatter([node.x for node in nodes if node.d == d], 
                [node.y for node in nodes if node.d == d], 
                    c=c_dict[d], s=80, label=label_dict[d])

    node = nodes[0]
    circle1 = plt.Circle((node.x, node.y), MIN_OD_DIST, ls='--', 
                            fill=False)               
    ax.add_patch(circle1)
    ax.annotate(text='Pickup', xy=(node.x, node.y), xytext=(50,-50), 
                textcoords='offset points',
                arrowprops={'width':1, 'color':'black'})

    node = nodes[int(len(nodes) / 2)]
    ax.annotate(text='Delivery', xy=(node.x, node.y), xytext=(-75,50), 
                textcoords='offset points', 
                arrowprops={'width':1, 'color':'black'})
    ax.set_xlabel('x')
    ax.set_ylabel('y', rotation=0)
    plt.legend(loc='upper right')
    if save:
        plt.savefig('images/example_instance.pdf')
    plt.show()


def save_to_dataframe(nodes, C, seed):
    logging.info(f'Saving instance to dataframe')
    df = pd.DataFrame()
    df['Task ID'] = range(2 * C)
    df['X'] = [node.x for node in nodes]
    df['Y'] = [node.y for node in nodes]
    df['Pickup'] = [i * int(i < C) + (i - C) * int(i >= C) 
                    for i in range(2 * C)]
    df['Delivery'] = [(i + C) * int(i < C) + i * int(i >= C) 
                      for i in range(2 * C)]
    df['Demand'] = [node.d for node in nodes]
    df['Load_Size'] = [node.q for node in nodes] 
    df.to_pickle(f'./data/num_custs-{C}-seed-{seed}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate instances for the Shared Truckload Problem')
    parser.add_argument('C', type=int, help='number of customers in problem')
    parser.add_argument('seed', type=int, help='seed for random number generation')
    args = parser.parse_args()
    data_path = 'data'
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    image_path = 'images'
    if os.path.exists(image_path) is False:
        os.makedirs(image_path)
    data = generate_instance(args.C, args.seed)
    save_to_dataframe(data, args.C, args.seed)
##    for C in range(5, 25, 1):
##        for seed in range(100):
##            data = generate_instance(C, seed)
##            #plot_instance(data)
##            save_to_dataframe(data, C, seed)
