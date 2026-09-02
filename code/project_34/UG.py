
""" 
Ultimatum Game on Complex Networks with coevolving update rules.
# TODO: More description
"""

# Library imports
import networkx as nx 
import numpy as np
import time
import os
import h5py
import csv
from numba import njit
from joblib import Parallel, delayed

### GLOBAL CONSTANTS ############################################################################
# Mapping the update rule strings to integer codes 
# and vice versa
RULE_TO_INT = {"MOR": 0, "REP": 1, "UI": 2}
INT_TO_RULE = {v: k for k, v in RULE_TO_INT.items()}

#################################################################################################
### FUNCTION DEFINITIONS ########################################################################
#################################################################################################

# Network generation ############################################################################
def generate_mixed_network(N:int=1000, alpha:float=0.0, m:int=4, m0_frac:float=0.1) -> nx.Graph:
    """Generates network according to mixed model by Gómez-Gardeñes and Moreno (2006),
    which interpolates between ER and BA model through the heterogeneity parameter alpha.

    Args:
        N (int, optional): Number of nodes. Defaults to 1000.
        alpha (float, optional): Heterogeneity parameter, can have values from 0 (BA-graph) 
            to 1 (ER-graph). Defaults to 0.0.
        m (int, optional): Number of edges added at each growth step. Defaults to 6.
        m0_frac (float, optional): Fraction of total nodes to form the initial fully connected
            graph before the growing process. Defaults to 0.1.

    Returns:
        nx.Graph: Output networkx graph.
    """
    m0 = int(N*m0_frac)
    G = nx.complete_graph(m0)  # initialise fully connected network with m0 nodes
    U_idx = np.arange(m0, N) # indices of N-m0 initially unconnected nodes
    G.add_nodes_from(U_idx)  # add the N-m0 unconnected nodes

    E = G.number_of_edges() # keeping track manually instead of extracting from graph in loop
    degree = np.zeros(N, dtype=int) # keeping track of degrees in a numpy array instead
    degree[:m0] = m0-1  # degree of the m0 nodes in the initial complete graph

    node_indices = np.arange(N) # used for masking later

    # we'll have to go through all N-m0 unconnected nodes, each selected randomly
    # let's do this by shuffeling the index list and go through the nodes in the
    # obtained random order
    np.random.shuffle(U_idx)

    for idx in U_idx:
        # create m links:
        for _ in range(m):
            # TODO: catch cases with no initial edges etc!
            # (re)compute node probs:
            mask = node_indices != idx # mask to get all nodes except current one
            # with probability alpha select a uniformly random one of all nods
            # with probability 1-alpha select node via preferential attachment
            probs = (1.0-alpha)*degree[mask]/(2*E - degree[idx]) +  alpha*1/(N-1)
            node = np.random.choice(node_indices[mask], p=probs)
            G.add_edge(idx, node) # add edge to graph

            degree[idx] += 1 # update degrees
            degree[node] += 1
            E += 1  # increment total number of edges

    return G

# Update functions ###################################################################################
@njit(cache=True)
def MOR_update(node, nb_row_ptr, nb_col_indices, payoffs, strategies, update_rules):
    """MORAN-like update rule: Choose one of the neighbors of the given node (or the node itself),
    with a probability proportional to their respective payoffs. Return the strategy and the update
    rule of that neighbor.

    Args:
        node (int): Node to update.
        nb_row_ptr (np.ndarray): CSR-style row pointers for the neighbors of the node. 
        nb_col_indices (np.ndarray): CSR-style column indices for the neighbors of the node.
        payoffs (np.ndarray): Array of all node payoffs.
        strategies (np.ndarray)): Array of all node strategies (each strategy consisting of offer and acceptence threshold).
        update_rules (np.ndarray): Array of all node update rules (encoded as integer).

    Returns:
        np.ndarray, int: New strategy and update rule for the given node.
    """

    start, end = nb_row_ptr[node], nb_row_ptr[node+1] # marking the index range with node's neighbors
    neighbors = nb_col_indices[start:end] # get the neighbors
    n_neighbors = neighbors.shape[0]      

    # aggregate total payoff 
    total = payoffs[node] # need to include the node itself
    for i in range(n_neighbors):
        total += payoffs[neighbors[i]]
    
    if total <= 0: # so if no payoff has been obtained at all
        # make uniformly random choice between all neighbors
        idx = np.random.randint(0, n_neighbors+1)
        if idx == n_neighbors:
            nb_chosen = node
        else:
            nb_chosen = neighbors[idx]
    else:
        # choose neighbor with probability proportional to payoff
        # instead of the weighted np.random.choice, let's do this the old fashioned way
        # via inverse cdf sampling
        r = np.random.random() * total  # generate uniform random var
        cumsum = payoffs[node]
        if r < cumsum: # first check for the node itself
            nb_chosen = node
        else:
            # as a fallback if we somehow get r so close to 1 that for some ungodly reason
            # the r < cumsum doesn't trigger in the last loop
            nb_chosen = neighbors[-1] 
            for i in range(n_neighbors):
                cumsum += payoffs[neighbors[i]]
                if r < cumsum: # we chose our node
                    nb_chosen = neighbors[i]
                    break
    # return strategy and update rule of chosen neighbor (or self)
    return strategies[nb_chosen], update_rules[nb_chosen]

@njit(cache=True)
def REP_update(node, nb_row_ptr, nb_col_indices, payoffs, strategies, update_rules, degrees):
    """Replicator update rule (corresponding to "Natural selection" rule): Choose a (uniformly)
    random neighbor of the given node and compare payoffs. If the neighbor's payoff is higher,
    choose the neighbor's strategy and update rule with a probability proportional to the difference
    in payoffs (otherwise keep the node's original strategy & update rule).

    Args:
        node (int): Node to update.
        nb_row_ptr (np.ndarray): CSR-style row pointers for the neighbors of the node. 
        nb_col_indices (np.ndarray): CSR-style column indices for the neighbors of the node.
        payoffs (np.ndarray): Array of all node payoffs.
        strategies (np.ndarray)): Array of all node strategies (each strategy consisting of offer and acceptence threshold).
        update_rules (np.ndarray): Array of all node update rules (encoded as integer).
        degrees (np.array): Array of all node degrees.

    Returns:
        np.ndarray, int: New strategy and update rule for the given node.
    """
    # corresponds to the "natural selection" rule
    start, end = nb_row_ptr[node], nb_row_ptr[node+1] # marking the index range with node's neighbors
    neighbors = nb_col_indices[start:end] # get the neighbors
    n_neighbors = neighbors.shape[0]      

    # choose random neighbor nb
    idx = np.random.randint(0, n_neighbors)
    nb = neighbors[idx]

    # compute replication probability from payoff difference
    prob = max(0.0, (payoffs[nb] - payoffs[node])/(2*max(degrees[node], degrees[nb])))

    if np.random.random() < prob: # copy nb
        return strategies[nb], update_rules[nb]
    else: # keep current strategy/update rule
        return strategies[node], update_rules[node]


@njit(cache=True)
def UI_update(node, nb_row_ptr, nb_col_indices, payoffs, strategies, update_rules):
    """Unconditional imitation update rule: For the given node, choose the neighbor with the highest payoff and compare.
    If the neighbor's payoff is higher, choose the neighbor's strategy and update rule, otherwise keep the node's
    original strategy and update rule.

    Args:
        node (int): Node to update.
        nb_row_ptr (np.ndarray): CSR-style row pointers for the neighbors of the node. 
        nb_col_indices (np.ndarray): CSR-style column indices for the neighbors of the node.
        payoffs (np.ndarray): Array of all node payoffs.
        strategies (np.ndarray)): Array of all node strategies (each strategy consisting of offer and acceptence threshold).
        update_rules (np.ndarray): Array of all node update rules (encoded as integer).

    Returns:
        np.ndarray, int: New strategy and update rule for the given node.
    """
    # get neighbor with highest payoff and copy if better
    start, end = nb_row_ptr[node], nb_row_ptr[node+1] # marking the index range with node's neighbors
    neighbors = nb_col_indices[start:end] # get the neighbors
    n_neighbors = neighbors.shape[0]  

    best_nb = node
    best_payoff = payoffs[node]
    for i in range(n_neighbors):
        nb = neighbors[i]
        if payoffs[nb] > best_payoff:
            best_payoff = payoffs[nb]
            best_nb = nb

    return strategies[best_nb], update_rules[best_nb]

# UG simulation functions #########################################################################
def init_UG(N, update_rules, update_ratios, mode:str="fair"):
    """Initialise the Ultimatum game with random strategies and update rules for each node, according to the specified
    player mode and update ratios. 
    Possible update rules are "REP" (Replicator update rule), "MOR" (Moran-like update rule) and "UI" (unconditional 
    imitation update rule). The update ratios specify the fraction each update rule should have in the initial population. 
    The player mode determines the strategies (which consist of offer p and acceptence threshold q for each player). The offer
    p is chosen as uniformly random value in [0.0, 1.0). The threshold q then depends on the mode: It can be "fair" (p = q),
    "pragmatic" (p=1-q) or "independent" (q is drawn independently as uniformly random value in [0.0, 1.0)).
    Returns the list of initial payoffs (all set to 0.0), strategies, and update rules for each node/player.


    Args:
        N (int): Number of nodes(players) in the underlying network.
        update_rules (list(str)): Update rules for players. 
        update_ratios (list or np.ndarray): Ratios with which each update rule occurs in the initial population. 
                        Must have same length as update_rules.
        mode (str, optional): Player mode, determining strategies as described above. Must be "fair", "pragmatic" or "independent". Defaults to "fair".

    Raises:
        ValueError: Invalid mode argument.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): initial payoffs, initial strategies, initial update rules of all nodes.
    """
    # initialise the ultimatum game by assigning initial strategies, update rules
    # and payoffs for each of the nodes in the graph
    
    # generate strategies
    if mode == "fair":  # p = q
        p_vals = np.random.uniform(size=(N, 1)) 
        strategies = np.column_stack([p_vals, p_vals])
    elif mode == "pragmatic":  # p = 1-q 
        p_vals = np.random.uniform(size=(N, 1))
        strategies = np.column_stack([p_vals, 1-p_vals])
    elif mode == "independent":  # p, q independent
        strategies = np.random.uniform(size=(N, 2))
    else: # invalide mode
        raise ValueError(f"Invalid mode argument. Mode must be 'fair', 'pragmatic', or 'independent'. Given mode: {mode}.")
    # make dictionary with node indices to assign it as node attribute

    # generate update rules
    # we want to randomly assign them to nodes, with the proportions specified by update_ratios
    # or at least as closely as we can
    update_ratios = np.array(update_ratios, dtype=float)
    update_ratios /= sum(update_ratios) # ensure they're normalized
    update_counts_raw = N*update_ratios # figure out the corresponding counts
    update_counts = np.floor(update_counts_raw).astype(int) # round down to integers 
    remaining_slots = N - update_counts.sum() # check how many (if any) slots are missing through the rounding
    if remaining_slots > 0: # increase counts of those with the biggest difference if necessary
        update_counts[np.argsort(update_counts_raw-update_counts)[:N - update_counts.sum()]] += 1
    update_keys = np.repeat(update_rules, update_counts) # generate array of update rules
    np.random.shuffle(update_keys) # shuffle for random assignment

    # return initial strategies, update rules, payoff
    return np.zeros(N, dtype=float), strategies, update_keys

@njit(cache=True)
def run_UG_step(src, trg, nb_row_ptr, nb_col_indices, degrees, strategies_curr, updates_curr):
    """Runs one round of the Ultimatum game, which consists of 2 phases:
        1. Playing: Each node playing against all its neighbors, once as proposer, once as responder and accumulating payoff
        2. Updating: Each node updating its strategy and update rule according to the accumulated payoffs and its update rule

    Args:
        src (np.ndarray):  Source node indices for all network edges.
        trg (np.ndarray): Target node indices for all network edges.
        nb_row_ptr (np.ndarray): Row pointers of the CSR-style sparse adjacency matrix representation of the network. 
        nb_col_indices (np.ndarray): Column indices of the CSR-style sparse adjacency matrix representation of the network. 
        degrees (np.ndarray): All node degrees.
        strategies_curr (np.ndarray): Current strategies (offer, acceptence threshold) of all nodes.
        updates_curr (np.ndarray): Current update rules of all nodes.

    Returns:
        payoffs (np.ndarray), strategies_new (np.ndarray), update_rules_new (np.ndarray): Accumulated payoffs, updated strategies and update rules of all nodes.
    """
    # Perform one simulation step, consisting of accumulating payoff and updating strategies/rules.
    # Each node "plays" each neighbor by comparing the respective offers p and thresholds q and 
    # adding the obtained payoff from each game. Then, according to each node's update rule, the 
    # payoffs are compared and strategies/update rules are updated
    N = len(degrees)
    # initialise payoffs to 0.0
    payoffs = np.zeros(N, dtype=float)

    # --------- Playing: Payoff resolution ------------
    # Note: right now we do not make use of knowing the player strategy (fair/pragmatic) to compute the
    # payoffs, for the sake of simplicity. Might be something to add later. 
    # go through all edges and resolve the pairwise interactions 
    # p = strategies_curr[:, 0]
    # q = strategies_curr[:, 1]
    # offer_accepted_r1 = p[src] >= q[trg] # round 1
    # offer_accepted_r2 = p[trg] >= q[src] # round 2 (switched roles)

    # accumulate the payoff 
    # round 1: src is proposer, trg is responder
    # np.add.at(payoffs, src[offer_accepted_r1], 1-p[src[offer_accepted_r1]]) # proposer payoff
    # np.add.at(payoffs, trg[offer_accepted_r1], p[src[offer_accepted_r1]]) # responder payoff
    # # round 2: trg is proposer, src is responder (roles switched)
    # np.add.at(payoffs, trg[offer_accepted_r2], 1-p[trg[offer_accepted_r2]]) # proposer payoff
    # np.add.at(payoffs, src[offer_accepted_r2], p[trg[offer_accepted_r2]]) # responder payoff
    # ... god I hope I have this the right way around
    
    for e in range(src.shape[0]): # iterate over edges
        i = src[e]
        j = trg[e]
        # get offer p/threshold q for nodes i, j
        p_i, q_i = strategies_curr[i]
        p_j, q_j = strategies_curr[j]
        # compute & update payoffs
        if p_i >= q_j:  # i proposes, j responds
            payoffs[i] += 1-p_i
            payoffs[j] += p_i
        if p_j >= q_i:  # j proposes, i responds
            payoffs[i] += p_j
            payoffs[j] += 1-p_j

    # --------- Updating: Update strategies & rules ---------
    # get lists with new strategies/update rules for each node
    strategies_new = np.empty_like(strategies_curr)
    update_rules_new = np.empty_like(updates_curr)

    for i in range(N):
        update = updates_curr[i]
        if update == 0: # MOR
            strategy_new, update_rule_new = MOR_update(i, nb_row_ptr, nb_col_indices, payoffs, strategies_curr, updates_curr)
        elif update == 1: # REP
            strategy_new, update_rule_new = REP_update(i, nb_row_ptr, nb_col_indices, payoffs, strategies_curr, updates_curr, degrees)
        elif update == 2: # UI
            strategy_new, update_rule_new = UI_update(i, nb_row_ptr, nb_col_indices, payoffs, strategies_curr, updates_curr)

        strategies_new[i] = strategy_new
        update_rules_new[i] = update_rule_new

    return payoffs, strategies_new, update_rules_new


def run_UG_simulation(file,
                      N=1000,
                      alpha=0.0,
                      m=4,
                      m0_frac=0.01,
                      num_rounds=1000, 
                      update_rules=["REP"], 
                      update_ratios=[1],
                      mode="independent",
                      snapshot_frequency=100,
                      rep_id=0, 
                      ):
    """Runs the Ultimatum game simulation with coevolving strategies and update rules for a given set of parameters and number of rounds.
    Generates a network with the given size and parameters, initialises the strategies and update rules of all nodes/players with the specified
    modes/distribution, and runs num_rounds Ultimatum Game steps. The state of the simulation (in terms of strategies, updates, payoffs) is 
    extracted and saved every snapshot_frequency steps. The extracted states are written as a group with datasets "steps", "p", "q", "updates", 
    "payoffs" to the given HDF5 file f. 

    Args:
        file (H5PY.File): HDF5 file to which the results are written.
        N (int, optional): Number of nodes/players. Defaults to 1000.
        alpha (float, optional): Network heterogeneity parameter (see generate_mixed_networks()). Defaults to 0.0.
        m (int, optional): Number of links created at each network growth step (see generate_mixed_networks()). Defaults to 4.
        m0_frac (float, optional): Fraction of nodes to from the initial complete network in the generation process
                        (see generate_mixed_networks()). Defaults to 0.01.
        num_rounds (int, optional): Number of rounds over which the UG evolves. Defaults to 1000.
        update_rules (list, optional): Update rules for players. Defaults to ["REP"].
        update_ratios (list, optional): Ratios with which each update rule occurs in the initial population. 
                        Must have same length as update_rules. Defaults to [1].
        mode (str, optional): Player mode determining the strategy generation. Must be "fair", "pragmatic", or "independent". 
                        Defaults to "independent".
        snapshot_frequency (int, optional): Determines after how many rounds a snapshot of the simulation state is extracted and saved. 
                        Defaults to 100.
        rep_id (int, optional): ID of the repetition, in case simulations are run repeatedly for a given combination of parameters.
                        Defaults to 0.
    """
    # start timer
    t0 = time.perf_counter()
    # initialize graph
    G = generate_mixed_network(N=N, alpha=alpha, m=m, m0_frac=m0_frac)
    # initialize UG    
    payoffs, strategies, updates = init_UG(N, update_rules, update_ratios, mode=mode)  
    updates = np.array([RULE_TO_INT[u] for u in updates], dtype=int) # use integers instead
    # TODO: INITIALISE UPDATES AS INTEGERS INSTEAD OF STRINGS to avoid the conversion later
    # extract graph structure that can be reused
    edge_arr = np.array(G.edges())
    src, trg = edge_arr[:, 0], edge_arr[:, 1]
    neighbor_matr = nx.to_scipy_sparse_array(G, format='csr', nodelist=range(N))
    row_ptr = neighbor_matr.indptr
    col_indices = neighbor_matr.indices
    degrees = np.diff(row_ptr)

    num_rounds_saved = num_rounds // snapshot_frequency
    # initialise result arrays
    res_steps = np.empty(num_rounds_saved, dtype=np.uint16)
    res_payoffs = np.empty((num_rounds_saved, N), dtype=np.float32)
    res_p = np.empty((num_rounds_saved, N), dtype=np.float32)
    res_q = np.empty((num_rounds_saved, N), dtype=np.float32)
    res_updates = np.empty((num_rounds_saved, N), dtype=np.uint8)

    # loop over UG rounds
    for r in range(num_rounds):
        # run UG
        payoffs, strategies, updates = run_UG_step(src, trg, row_ptr, col_indices, degrees, strategies, updates)

        # extract result data every couple of rounds to see evolution
        if r%snapshot_frequency == 0:
                round_idx = round//snapshot_frequency
                res_steps[round_idx] = round_idx
                res_payoffs[round_idx, :] = payoffs
                res_p[round_idx, :] = strategies[:, 0]
                res_q[round_idx, :] = strategies[:, 1]
                res_updates[round_idx, :] = updates

    # tracking the runtime
    t1 = time.perf_counter() - t0
    # write results to file
    grp = file.create_group(f"rep_{rep_id}")
    grp.create_dataset("steps", data=res_steps)
    grp.create_dataset("payoffs", data=res_payoffs)
    grp.create_dataset("p", data=res_p)
    grp.create_dataset("q", data=res_q)
    grp.create_dataset("updates", data=res_updates)

    grp.attrs["rep_id"] = rep_id
    grp.attrs["runtime_s"] = t1


def run_UG_combo(alpha,
                 mode,
                 update_rules,
                 update_ratios,
                 N=1000,
                 m=4,
                 m0_frac=0.01,
                 num_reps=100,
                 num_rounds=1000, 
                 snapshot_frequency=100,
                 output_path="results/" 
                 ):
    """Runs num_reps repeated simulations of the Ultimatum Game with coevolving update rules for a given combination of parameters 
    (network heterogeneity alpha, player mode, update rules, initial update ratios) by repeatedly calling run_UG_simulation(). 
    Results are written to an HDF5 file at the specified output_path, with each repetition being saved as a group.

    Args:
        alpha (float): Network heterogeneity parameter (see generate_mixed_networks()).
        mode (str): Player mode determining the strategy generation. Must be "fair", "pragmatic", or "independent".
        update_rules (list): Update rules for players. 
        update_ratios (list): Ratios with which each update rule occurs in the initial population. 
                        Must have same length as update_rules.
        N (int, optional): Number of nodes/players. Defaults to 1000.
        m (int, optional): Number of links created at each network growth step (see generate_mixed_networks()). Defaults to 4.
        m0_frac (float, optional):Fraction of nodes to from the initial complete network in the generation process
                        (see generate_mixed_networks()). Defaults to 0.01.
        num_reps (int, optional): Number of simulation repetitions. Defaults to 100.
        num_rounds (int, optional): Number of rounds over which the UG evolves in each simulation. Defaults to 1000.
        snapshot_frequency (int, optional): Determines after how many rounds a snapshot of the simulation state is extracted and saved. 
                        Defaults to 100.
        output_path (str, optional): Path where result file will be saved. Defaults to "results/".


    Returns:
        dict: Dict containing filename, parameters, and simulation status (ok, skipped if file already existed, or error if it failed).
    """
    # start timer
    t0 = time.perf_counter()
    filename = output_path + f"a{int(alpha*100)}_{mode}_{int(update_ratios[0]*100)}{update_rules[0]}_vs_{int(update_ratios[1]*100)}{update_rules[1]}.hdf5"

    try:
        # create & open file
        with h5py.File(filename, "x") as f:
            # set sim parameters as attributes
            f.attrs["alpha"] = alpha
            f.attrs["mode"] = mode
            f.attrs["update_rules"] = update_rules
            f.attrs["update_ratio"] = update_ratios
            f.attrs["num_reps"] = num_reps
            f.attrs["N"] = N
            f.attrs["snapshot_frequency"] = snapshot_frequency
            f.attrs["num_rounds"] = num_rounds

            # run desired number of repetitions of sims
            for i in range(num_reps):
                run_UG_simulation(
                    file=f,                
                    alpha=alpha,
                    update_rules=update_rules,
                    update_ratios=update_ratios,
                    mode=mode,
                    N=N,
                    m=m,
                    m0_frac=m0_frac,
                    num_rounds=num_rounds,
                    snapshot_frequency=snapshot_frequency,
                    rep_id=i,
                )

            t1 = time.perf_counter() - t0
            f.attrs["runtime"] = t1

            return {
                "filename": filename,
                "alpha": alpha,
                "mode": mode,
                "update_rules": update_rules,
                "update_ratios": update_ratios,
                "status": "ok",
                "runtime": t1}
            
    except FileExistsError:
        return {"filename": filename,
                "alpha": alpha,
                "mode": mode,
                "update_rules": update_rules,
                "update_ratios": update_ratios,
                "status": "exists_skipped",
                "runtime": None}

    except Exception as e:
        return {"filename": filename,
                "alpha": alpha,
                "mode": mode,
                "update_rules": update_rules,
                "update_ratios": update_ratios,
                "status": "failed",
                "error": str(e),
                "runtime": None}


def run_parameter_sweep(alphas,
                        modes,
                        update_combos, 
                        update_ratios, 
                        N=1000,
                        m=4,
                        m0_frac=0.01,
                        num_rounds=1000,
                        output_path="results/",
                        num_reps=100, 
                        snapshot_frequency=100
                        ):
    """Run a parameter sweep of simulations of the Ultimatum game (UG) with coevolving update rules on complex networks. For each 
    combination of network heterogeneity parameters alphas, player modes, update rule combinations, and initial update rule ratios,
    num_reps repetitions of the UG simulations are run. 
    A single UG simulation consists of:
        1. generating a network according to the specified parameters (via generate_mixed_networks()), 
        2. initialising the initial strategies and update rules,
        3. then evolving for num_rounds total by iteratively: 
            - playing the game, accumulating payoff and
            - updating the strategies and update rules of the nodes/players. 
    The simulations for different combinations are parallelized via joblib (calling run_UG_combo() for each cobination, which performs the specified number of 
    repetitions sequentially). 
    The simulation results of each parameter combination are written to its own HDF5 file, all of of which are saved in the 
    directory specified via output_path. Each file contains the results of each simulation repetition as a group (so num_reps 
    groups per file). Furthermore, a csv file is produced in the end (sim_overview.csv), detailing the result file names, the 
    corresponding parameter combinations, and the success/fail status of each combination run.

    Args:
        alphas (list or np.ndarray): Network heterogeneity parameters to test (see generate_mixed_network()).
        modes (list): Player modes determining strategies to test. Supports "fair", "pragmatic", and "independent".
        update_combos (list): Update rule combinations to test. Supports rule pairs only. Rules can be "REP", "MOR", or "UI".
        update_ratios (list or np.ndarray): Ratios with which each update rule occurs in the initial population. 
                        Each element must have same length as each element in update_rules.
        N (int, optional): Number of nodes/players. Defaults to 1000.
        m (int, optional): Number of links created at each network growth step (see generate_mixed_networks()). Defaults to 4.
        m0_frac (float, optional):Fraction of nodes to from the initial complete network in the generation process
                        (see generate_mixed_networks()). Defaults to 0.01.
        num_rounds (int, optional): Number of rounds over which the UG evolves in each simulation. Defaults to 1000.
        output_path (str, optional): Path where result file will be saved. Defaults to "results/".
        num_reps (int, optional): Number of simulation repetitions. Defaults to 100.
        snapshot_frequency (int, optional): Determines after how many rounds a snapshot of the simulation state is extracted and saved. 
                        Defaults to 100.         
    """

    os.makedirs(output_path, exist_ok=True)

    # Big loop of doom over parameter combinations
    combos = [dict(alpha=a, mode=m, update_rules=update_combo, update_ratios=ratio) 
              for a in alphas for m in modes for update_combo in update_combos
              for ratio in update_ratios]

    # run simulations (with joblib)
    results = Parallel(n_jobs=-1)(
        delayed(run_UG_combo)(
            **c,
            N=N,
            m=m,
            m0_frac=m0_frac,
            num_rounds=num_rounds,
            snapshot_frequency=snapshot_frequency,
            num_reps=num_reps,
            output_path=output_path)
            for c in combos
        )

    # write results to a sim overview csv file
    fields = ("filename", "alpha", "mode", "update_rules", "update_ratios", "status", "runtime", "error")
    with open(output_path+"sim_overview.csv", 'w', newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(results)

#################################################################################################
### MAIN ########################################################################################
#################################################################################################

def main():
    # CONSTANTS AND SIMULATION PARAMETERS
    N = 1000  # number of nodes/players 
    NUM_ROUNDS = 1000  # Number of rounds for evolving the UG simulations
    SNAPSHOT_FREQUENCY = 1  # How often to save simulation state
    NUM_REPS = 100  # Number of simulation repetitios for each combination
    M = 4  # Number of new connections per node in the network growth process
    M0_FRAC = 0.01  # Fraction of nodes forming the initial complete network in the network generation process
    OUTPUT_PATH = "results/"  # Where to save the result files
    ALPHAS = [0.0, 0.5, 1.0]  # Network heterogeneity values
    MODES = ["fair", "pragmatic", "independent"]  # player modes determining the strategies
    UPDATE_COMBOS = [["REP", "UI"], ["MOR", "REP"], ["UI", "MOR"]]  # update rule combinations
    FRACS = np.linspace(0.0, 1.0, 21)  
    UPDATE_RATIOS = np.column_stack((FRACS, 1.0-FRACS))  # initial ratios of update rules

    # print sweep info to check
    print("Starting parameter sweep!")
    print(f"\nRunning {len(ALPHAS)*len(MODES)*len(UPDATE_COMBOS)*len(UPDATE_RATIOS)} parameter combinations, with {NUM_REPS} each.")
    print("alpha values: ", ALPHAS)
    print("player modes: ", MODES)
    print("Update rule combinations: ", UPDATE_COMBOS)
    print("Ratios to test: ", UPDATE_RATIOS)
    print("Number of UG rounds: ", NUM_ROUNDS)

    # run sim
    run_parameter_sweep(ALPHAS, MODES, UPDATE_COMBOS, UPDATE_RATIOS,
                        N=N, m=M, m0_frac=M0_FRAC, num_rounds=NUM_ROUNDS, 
                        num_reps=NUM_REPS, output_path=OUTPUT_PATH, 
                        snapshot_frequency=SNAPSHOT_FREQUENCY )
    print("Done!")

if __name__ == "__main__":
    main()