from typing import Dict, List
from unittest.result import failfast

from numpy import ndarray
import numpy as np

from factor import Factor

'''
########## READ CAREFULLY #############################################################################################
You should implement all the functions in this file. Do not change the function signatures.
#######################################################################################################################
'''
        
    
def restrict(factor: Factor, variable: str, value: int) -> Factor:
    '''
    Restrict a factor by assigning value to variable
    :param factor: a Factor object
    :param variable: the name of the variable to restrict
    :param value: the value to restrict variable to
    :return: a new Factor object resulting from restricting variable. This factor no longer includes variable in its
             var_list.
    '''

    '''
    ##### YOUR CODE HERE #####
    '''
    index_of_var = factor.var_list.index(variable)
    factor.var_list.remove(variable)
    new_values = np.moveaxis(factor.values,index_of_var,0)
    new_values = new_values[value]
    return Factor(factor.var_list,new_values)

    
def sortByOrder(values:ndarray, varList:List[str], sortingOrder:List[str]):
    for i in range(len(sortingOrder)):
        var = sortingOrder[i]
        if (varList[i] == var):
            continue
        trueIndex = varList.index(var)
        values = values.swapaxes(i, trueIndex)
        varList[trueIndex] = varList[i]
        varList[i] = var
        
    return values,varList

def sameVar(varlist_a: List[str], varlist_b: List[str]):
    return set(varlist_a) & set(varlist_b)
    
def recursionMulti(a_values: ndarray, b_values: ndarray, a_varList:List[str], b_varList:List[str]):
    # if (len(a_varList) == 0 or len(b_varList == 0)):
    #     return np.multiply.outer()    

    if (len(a_varList) == 0 or len(b_varList) == 0):
        
        final_values = a_values * b_values

        return final_values
   

    if (a_varList[0] == b_varList[0]):
        # value1 = recursionMulti(a_values[0], b_values[0], a_varList[1:], b_varList[1:])
        # values2 = recursionMulti(a_values[1], b_values[1], a_varList[1:], b_varList[1:])
        finalValues= []
        for i in range(a_values.shape[0]):
            value = recursionMulti(a_values[i], b_values[i], a_varList[1:], b_varList[1:])
            finalValues.append(value)


        
        final_values = np.array(finalValues)
        return final_values
    else:
        final_values = np.multiply.outer(a_values,b_values)

        return final_values

def multiply(factor_a: Factor, factor_b: Factor) -> Factor:
    '''
    Multiply two tests (factor_a and factor_b) together.
    :param factor_a: a Factor object representing the first factor in the multiplication operation
    :param factor_b: a Factor object representing the second factor in the multiplication operation
    :return: a new Factor object resulting from the multiplication of factor_a and factor_b. Note that the new factor's
             var_list is the union of the var_lists of factor_a and factor_b IN ALPHABETICAL ORDER.
    '''

    '''
    ##### YOUR CODE HERE #####
    '''
    
    if(len(factor_a.var_list) == 0):
       
        return  Factor(factor_b.var_list, factor_b.values*factor_a.values[0])
    if(len(factor_b.var_list) == 0):
        return  Factor(factor_a.var_list, factor_a.values*factor_a.values[0])
    sameVarList = list(sameVar(factor_a.var_list, factor_b.var_list))
    sameVarList.sort()
    a_values,a_var_list = sortByOrder(factor_a.values, factor_a.var_list, sameVarList)

    b_values,b_var_list = sortByOrder(factor_b.values, factor_b.var_list, sameVarList)
    final_values = recursionMulti(a_values,b_values, a_var_list, b_var_list)
   
    final_vars = a_var_list+ b_var_list[len(sameVarList):]
    final_vars_sort = final_vars.copy()
    final_vars_sort.sort()
    
    final_values,final_vars = sortByOrder(final_values,final_vars,final_vars_sort)

    
    new_factor = Factor(final_vars, final_values)
    return new_factor


def sum_out(factor: Factor, variable: str) -> Factor:
    '''
    Sum out a variable from factor.
    :param factor: a Factor object
    :param variable: the name of the variable in factor that we wish to sum out
    :return: a Factor object resulting from performing the sum out operation on factor. Note that this new factor no
             longer includes variable in its var_list.
    '''

    '''
    ##### YOUR CODE HERE #####
    '''
    index_of_var = factor.var_list.index(variable)
    new_values = np.moveaxis(factor.values, index_of_var, -1)
    
    factor.var_list.remove(variable)
  
    

    new_values = np.add.reduce(new_values.transpose(),0).transpose()
    
    new_factor = Factor(factor.var_list, new_values)

    return new_factor


def normalize(factor: Factor) -> Factor:
    '''
    Normalize factor such that its values sum to 1.
    :param factor: a Factor object representing the factor to normalize
    :return: a Factor object resulting from performing the normalization operation on factor
    '''

    '''
    ##### YOUR CODE HERE #####
    '''
    currentSum = factor.values.sum()
    new_values = factor.values / currentSum
    new_factor = Factor(factor.var_list, new_values)
    
    return new_factor

        
def vea(factor_list: List[Factor], query_variables: List[str], evidence: Dict[str, int], ordered_hidden_variables: List[str], verbose: bool=False) -> (Factor, int):
    '''
    Applies the Variable Elimination Algorithm for input tests factor_list, restricting tests according to the
    evidence in evidence_list, and eliminating hidden variables in the order that they appear in
    ordered_list_hidden_variables. The result is the distribution for the query variables. The query variables are, by
    process of elimination, those for which we do not have evidence for and do not appear in the list of hidden
    variables).
    :param factor_list: a list of Factor objects representing every conditional probability distribution in the
                        Bayesian network
    :param query_variables: a list of variable names corresponding to the query variables
    :param evidence_list: a dict mapping evidence variable names to corresponding values
    :param ordered_list_hidden_variables: a list of names of the hidden variables. Variables are to be eliminated in the
                                          order that they appear in this list.
    :param verbose: Whether to print results of intermediate VEA operations (use for debugging, if you like)
    :return: A Factor object representing the result of executing the Variable Elimination Algorithm for the given
             evidence and ordered list of hidden variables, treewidth of the network
    '''

    '''
    ##### YOUR CODE HERE #####
    '''   
    # restriction
    for i in range(len(factor_list)):
        for var in factor_list[i].var_list.copy():
            if var in evidence.keys():
                factor_list[i] = restrict(factor_list[i], var, evidence.get(var))
                
        
    # elimination
    treewidth = 0
    for hiddenVar in ordered_hidden_variables:
        newF = Factor([], np.array([1]))
        for factor in factor_list.copy():
            
            if factor.has_variable(hiddenVar):
                newF = multiply(newF,factor)
                factor_list.remove(factor)
        
        newF = sum_out(newF,hiddenVar)

        if (newF.n_vars > treewidth):
            treewidth = newF.n_vars
        factor_list.append(newF)
        
    
    result = Factor([], np.array([1]))


    for factor in factor_list:

        result = multiply(result,factor)

   
    result = normalize(result)

    return result, treewidth


