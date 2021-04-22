# -*- coding: utf-8 -*-
"""
@author: akram_iea
"""
import os
import sys
import numpy as np
import json


####################
# Utils ##########
####################

# Global variables
CUR_DIR = os.getcwd()
PAR_DIR = os.path.dirname(CUR_DIR)
RESOURCES_STR = 'Resources'
SEASONS_STR = 'Seasons'
INTERVENTIONS_STR = 'Interventions'
EXCLUSIONS_STR = 'Exclusions'
T_STR = 'T'
SCENARIO_NUMBER = 'Scenarios_number'
RESOURCE_CHARGE_STR = 'workload'
TMAX_STR = 'tmax'
DELTA_STR = 'Delta'
MAX_STR = 'max'
MIN_STR = 'min'
RISK_STR = 'risk'
START_STR = 'start'
QUANTILE_STR = "Quantile"
ALPHA_STR = "Alpha"

# Json reader


def read_json(filename: str):
    """Read a json file and return data as a dict object"""

    print('Reading json file ' + filename + '...')
    file = open(filename, 'r')
    instance = json.load(file)
    f.close()
    print('Done')

    return instance

# Txt Solution reader


def read_solution_from_txt(instance: dict, solution_filename: str):
    """Read a txt formated Solution file, and store the solution informations in Instance"""

    print('Loading solution from ' + solution_filename + '...')
    # Load interventions
    interventions = instance[INTERVENTIONS_STR]
    # Read file line by line, and store starting time value (no checks yet, except format and duplicate)
    solution_file = open(solution_filename, 'r')
    for line in solution_file:
        # Split line to retrive infos: Intervention name and decided starting date
        tmp = line.split(' ')
        intervention_name = tmp[0]
        start_time_str = tmp[1].split('\n')[0]
        # Assert Intervention exists
        if not intervention_name in interventions:
            print('ERROR: Unexpected Intervention ' + intervention_name + ' in solution file ' + solution_filename + '.')
            continue
        # Assert starting date is an integer
        start_time: int
        try:
            start_time = int(start_time_str)
        except ValueError:
            print('ERROR: Unexpected starting time ' + start_time_str + ' for Intervention ' + intervention_name + '. Expect integer value.')
            continue
        # Assert no duplicate
        if START_STR in interventions[intervention_name]:
            print('ERROR: Duplicate entry for Intervention ' + intervention_name + '. Only first read value is being considered.')
            continue
        # Store starting time
        interventions[intervention_name][START_STR] = start_time
    solution_file.close()
    print('Done')


################################
# Results processing ##########
################################

# Compute effective worload
def compute_resources(instance: dict):
    """Compute effective workload (i.e. resources consumption values) for every resource and every time step"""

    # Retrieve usefull infos
    Interventions = instance[INTERVENTIONS_STR]
    T_max = instance[T_STR]
    Resources = instance[RESOURCES_STR]
    # Init resource usage dictionnary for each resource and time
    resources_usage = {}
    for resource_name in Resources.keys():
        resources_usage[resource_name] = np.zeros(T_max)
    # Compute value for each resource and time step
    for intervention_name, intervention in Interventions.items():
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = start_time - 1
        # index of list starts at 0
        intervention_worload = intervention[RESOURCE_CHARGE_STR]
        intervention_delta = int(intervention[DELTA_STR][start_time_idx])
        # compute effective worload
        for resource_name, intervention_resource_worload in intervention_worload.items():
            for time in range(start_time_idx, start_time_idx + intervention_delta):
                # null values are not available
                if str(time+1) in intervention_resource_worload and str(start_time) in intervention_resource_worload[str(time+1)]:
                    resources_usage[resource_name][time] += intervention_resource_worload[str(time+1)][str(start_time)]

    return resources_usage


# Retrieve effective risk distribution given starting times solution
def compute_risk_distribution(Interventions: dict, T_max: int, scenario_numbers):
    """Compute risk distributions for all time steps, given the interventions starting times"""

    print('\tComputing risk...')
    # Init risk table
    risk = [scenario_numbers[t] * [0] for t in range(T_max)]
    # Compute for each intervention independently
    for intervention in Interventions.values():
        # Retrieve Intervention's usefull infos
        intervention_risk = intervention[RISK_STR]
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = int(start_time) - 1 # index for list getter
        delta = int(intervention[DELTA_STR][start_time_idx])
        for time in range(start_time_idx, start_time_idx + delta):
            for i, additional_risk in enumerate(intervention_risk[str(time + 1)][str(start_time)]):
                risk[time][i] += additional_risk
    print('\tDone')

    return risk

# Compute mean for each period


def compute_mean_risk(risk, T_max: int, scenario_numbers):
    """Compute mean risk values over each time period"""

    print('\tComputing mean risk...')
    # Init mean risk
    mean_risk = np.zeros(T_max)
    # compute mean
    for t in range(T_max):
        mean_risk[t] = sum(risk[t]) / scenario_numbers[t]
    print('\tDone')

    return mean_risk

# Compute quantile for each period


def compute_quantile(risk, T_max: int, scenario_numbers, quantile):
    """Compute Quantile values over each time period"""

    print('\tComputing Quantile...')
    # Init quantile
    q = np.zeros(T_max)
    for t in range(T_max):
        risk[t].sort()
        q[t] = risk[t][int(np.ceil(scenario_numbers[t] * quantile))-1]
    print('\tDone')

    return q

# Compute both objectives: mean risk and quantile


def compute_objective(Instance: dict):
    """Compute objectives (mean and expected excess)"""

    print('Computing objectives values...')
    # Retrieve usefull infos
    T_max = Instance[T_STR]
    scenario_numbers = Instance[SCENARIO_NUMBER]
    Interventions = Instance[INTERVENTIONS_STR]
    quantile = Instance[QUANTILE_STR]
    # Retrieve risk final distribution
    risk = compute_risk_distribution(Interventions, T_max, scenario_numbers)
    # Compute mean risk
    mean_risk = compute_mean_risk(risk, T_max, scenario_numbers)
    # Compute quantile
    q = compute_quantile(risk, T_max, scenario_numbers, quantile)
    print('Done')

    return mean_risk, q



##################################
# Constraints checkers ##########
##################################

## Launch each Constraint checks
def check_all_constraints(Instance: dict):
    """Run all constraint checks"""

    print('Checking constraints...')
    # Schedule constraints
    check_schedule(Instance)
    # Resources constraints
    check_resources(Instance)
    # Exclusions constraints
    check_exclusions(Instance)
    if check_exclusions(Instance) and check_resources(Instance) and check_schedule(Instance):
        print('Done')
        return(True)
    else:
        return(False)

# Schedule constraints: §4.1 in model description


def check_schedule(instance: dict):
    """Check schedule constraints"""

    print('\tChecking schedule constraints...')
    # Continuous interventions: §4.1.1
    #   This constraint is implicitly checked by the resource computation:
    #   computation is done under continuity hypothesis, and resource bounds will ensure the feasibility
    # Checks is done on each Intervention
    interventions = instance[INTERVENTIONS_STR]
    for intervention_name, intervention in interventions.items():
        # interventions are planned once: §4.1.2
        #   assert a starting time has been assigned to the intervention
        if not START_STR in intervention:
            print('ERROR: Schedule constraint 4.1.2: Intervention ' + intervention_name + ' has not been scheduled.')
            return False
        # Starting time validity: no explicit constraint
        start_time = intervention[START_STR]
        horizon_end = instance[T_STR]
        if not (1 <= start_time <= horizon_end):
            print('ERROR: Schedule constraint 4.1 time validity: Intervention ' + intervention_name + ' starting time ' + str(start_time)
            + ' is not a valid starting date. Expected value between 1 and ' + str(horizon_end) + '.')
            # Remove start time to avoid later access errors
            del intervention[START_STR]
            return False
        # No work left: §4.1.3
        #   assert intervention is not ongoing after time limit or end of horizon
        time_limit = int(intervention[TMAX_STR])
        if time_limit < start_time:
            print('ERROR: Schedule constraint 4.1.3: Intervention ' + intervention_name + ' realization exceeds time limit.'
            + ' It starts at ' + str(start_time) + ' while time limit is ' + str(time_limit) + '.')
            # Remove start time to avoid later access errors
            del intervention[START_STR]
            return False

    print('\tDone')
    return True

# Resources constraints: §4.2 in model description


def check_resources(instance: dict):
    """Check resources constraints"""
    print('\tChecking resources constraints...')
    t_max = instance[T_STR]
    resources = instance[RESOURCES_STR]
    # Bounds are checked with a tolerance value
    tolerance = 1e-5
    # Compute resource usage
    resource_usage = compute_resources(instance) # dict on resources and time
    # Compare bounds to usage
    for resource_name, resource in resources.items():
        for time in range(t_max):
            # retrieve bounds values
            upper_bound = resource[MAX_STR][time]
            lower_bound = resource[MIN_STR][time]
            # Consumed value
            worload = resource_usage[resource_name][time]
            # Check max
            if worload > upper_bound + tolerance:
                print('ERROR: resources constraint 4.2 upper bound: Worload on Resource ' + resource_name + ' at time ' + str(time+1) + ' exceeds upper bound.'
                + ' Value ' + str(worload) + ' is greater than bound ' + str(upper_bound) + ' plus tolerance ' + str(tolerance) + '.')
                return False
            # Check min
            if worload < lower_bound - tolerance:
                print('ERROR: resources constraint 4.2 lower bound: Worload on Resource ' + resource_name + ' at time ' + str(time+1) + ' does not match lower bound.'
                + ' Value ' + str(worload) + ' is lower than bound ' + str(lower_bound) + ' minus tolerance ' + str(tolerance) + '.')
                return False
    print('\tDone')
    return True

# Exclusions constraints: §4.3 in model description


def check_exclusions(Instance: dict):
    """Check exclusions constraints"""

    print('\tChecking exclusions constraints...')
    # Retrieve interventions and exclusions
    interventions = Instance[INTERVENTIONS_STR]
    exclusions = Instance[EXCLUSIONS_STR]
    # Assert every exclusion holds
    for exclusion in exclusions.values():
        # Retrieve exclusion infos
        [intervention_1_name, intervention_2_name, season] = exclusion
        # Retrieve concerned interventions...
        intervention_1 = interventions[intervention_1_name]
        intervention_2 = interventions[intervention_2_name]
        # start time should be defined (already checked in scheduled constraint checker)
        if (not START_STR in intervention_1) or (not START_STR in intervention_2):
            continue
        # ... their respective starting times...
        intervention_1_start_time = intervention_1[START_STR]
        intervention_2_start_time = intervention_2[START_STR]
        # ... and their respective deltas (duration)
        intervention_1_delta = int(intervention_1[DELTA_STR][intervention_1_start_time - 1]) # get index in list
        intervention_2_delta = int(intervention_2[DELTA_STR][intervention_2_start_time - 1]) # get index in list
        # Check overlaps for each time step of the season
        for time_str in Instance[SEASONS_STR][season]:
            time = int(time_str)
            if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                print('ERROR: exclusions constraint 4.3: interventions ' + intervention_1_name + ' and ' + intervention_2_name
                    + ' are both ongoing at time ' + str(time) + '.')
                return False
    print('\tDone')
    return True


#######################
# Displayer ##########
#######################

# Basic printing
def display_basic(Instance: dict, mean_risk, quantile):
    """Print main infos"""

    # Usefull infos
    alpha = Instance[ALPHA_STR]
    q = Instance[QUANTILE_STR]
    # Infos about instance
    print('Instance infos:')
    print('\tInterventions number: ', len(Instance[INTERVENTIONS_STR]))
    print('\tScenario numbers: ', len(Instance[SCENARIO_NUMBER]))
    # Computed infos about solution
    print('Solution evaluation:')
    # print('\tmean_risk over time: ', mean_risk)
    obj_1 = np.mean(mean_risk)
    print('\tObjective 1 (mean risk): ', obj_1)
    # print('\tQuantile (Q' + str(q) + ') over time: ', quantile)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    print('\tObjective 2 (expected excess  (Q' + str(q) + ')): ', obj_2)
    obj_tot = alpha * obj_1 + (1-alpha)*obj_2
    print('\tTotal objective (alpha*mean_risk + (1-alpha)*expected_excess): ', obj_tot)
    return obj_tot


######################
# Launcher ##########
######################

def check_and_display(instance_file, solution_file):
    """Control checker actions"""

    # Read Instance
    instance = read_json(instance_file)
    # Read Solution
    read_solution_from_txt(instance, solution_file)
    # Check all constraints
    check_all_constraints(instance)
    # Compute indicators
    mean_risk, quantile = compute_objective(instance)
    # Display Solution
    return display_basic(instance, mean_risk, quantile)


def obj_tot(solution_file):
    instance = read_json("example3.json")
    return check_and_display("example3.json", solution_file)


def neighboroperator(solution_file):
    instance = read_json("example3.json")
    # Read Solution
    read_solution_from_txt(instance, solution_file)
    Neighborhood = []
    start_position = []
    interventions_list = list(instance["Interventions"].keys())
    for i in interventions_list:
        start_position.append(instance["Interventions"][i]["start"])
    for j in interventions_list:
        for h in range(len(interventions_list)):
            instance["Interventions"][interventions_list[h]]["start"] = start_position[h]
        for i in range(1, 19):
            instance["Interventions"][j]["start"] = i
            if check_all_constraints(instance):
                Neighborhood.append([instance["Interventions"][u]["start"] for u in interventions_list])
    while start_position in Neighborhood:
        Neighborhood.pop(Neighborhood.index(start_position))
    neighborhood_files_names = []
    h= 2
    for _list in Neighborhood:
        with open('oooutput' + str(h) + '.txt', 'w') as f:
            neighborhood_files_names.append('oooutput' + str(h) + '.txt')

            for i, item in enumerate(_list):
                f.write(interventions_list[i] + " " +"%s\n" % item)
            f.close()
        h += 1
    return neighborhood_files_names


"""
Tabu Search Class
"""


class TabuSearch:
    def __init__(self, initialSolution, solutionEvaluator, neighborOperator,
                 acceptableScoreThreshold, tabuTenure):
        self.currSolution = initialSolution
        self.bestSolution = initialSolution
        self.evaluate = solutionEvaluator
        self.neighborOperator = neighborOperator
        self.acceptableScoreThreshold = acceptableScoreThreshold
        self.tabuTenure = tabuTenure

    def isTerminationCriteriaMet(self):
        # can add more termination criteria
        return self.evaluate(self.bestSolution) < self.acceptableScoreThreshold \
               or self.neighborOperator == []

    def run(self):
        tabuList = {}
        iter = 0
        solutions_amelioration = []

        while not self.isTerminationCriteriaMet():
            if iter == 2:
                print("debug")
            # get all of the neighbors
            neighbors = self.neighborOperator(self.currSolution)
            # find all tabu_solutions other than those
            # that fit the aspiration criteria
            tabu_solutions = tabuList.keys()
            # find all neighbors that are not part of the Tabu list
            neighbors = filter(lambda n: n not in tabu_solutions, neighbors)
            # pick the best neighbor solution
            new_solution = sorted(neighbors, key=lambda n: self.evaluate(n))[0]
            # get the cost between the two solutions
            cost = self.evaluate(self.bestSolution) - self.evaluate(new_solution)
            # if the new solution is better,
            # update the best solution with the new solution
            if cost >= 0:
                self.bestSolution = new_solution
            # update the current solution with the new solution
            self.currSolution = new_solution
            solutions_amelioration.append(self.evaluate(self.bestSolution))

            # decrement the Tabu Tenure of all tabu list solutions
            for sol in tabuList:
                tabuList[sol] -= 1
                if tabuList[sol] == 0:
                    del tabuList[sol]
            # add new solution to the Tabu list
            tabuList[new_solution] = self.tabuTenure
            iter +=1

        # return best solution found
        return self.bestSolution


if __name__ == '__main__':
    initialSolution = "output0.txt"
    solutionEvaluator = obj_tot
    neighborOperator = neighboroperator
    acceptableScoreThreshold = 1508
    tabuTenure = 5
    result = TabuSearch(initialSolution, solutionEvaluator, neighborOperator, acceptableScoreThreshold, tabuTenure)
    f = open(result.run(), "r")
    solution = f.read()
    print("the best solution is :\n" + solution)