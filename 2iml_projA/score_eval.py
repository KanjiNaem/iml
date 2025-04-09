import pandas as pd

def eval_score(): #doest work since sample_solution != reference_solution
    result_arr = pd.read_csv("results.csv").to_numpy()
    sample_arr = pd.read_csv("sample.csv").to_numpy()

    result_data_arr = []
    sample_data_arr = []
    
    for x in result_arr:
            result_data_arr.append(x[0])
    
    for x in sample_arr:
          sample_data_arr.append(x[0])


    if len(sample_data_arr) != len(result_data_arr):
          raise Exception("sample data array and result data array are of different length (bad, cringe!)") 

    semi_result_sum = 0
    for i in range(len(sample_data_arr)):
          semi_result_sum += (abs(result_data_arr[i] - sample_data_arr[i])) / sample_data_arr[i]

    result = semi_result_sum * 100  
    print(result)
    return result


eval_score()
