#include <boost/multiprecision/mpfi.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <tuple>
#include <chrono>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <iomanip>
#include "myUtility.h"

using namespace std;
using namespace std::chrono;
using namespace boost::multiprecision;

tuple<mpfi_float_50, int> myInterval(vector<vector<mpfi_float_50>> I_X0, int Pred_X0, vector<vector<long double>> W1, vector<vector<long double>> W2, vector<long double> b1, int inputs_size, int hidden_neural_num, int output_size)
{

    // convert to interval
    vector<vector<mpfi_float_50>> I_W1 = convert_2D_matrix(W1);
    vector<vector<mpfi_float_50>> I_W2 = convert_2D_matrix(W2);
    vector<mpfi_float_50> I_b1 = convert_array(b1);

    // init
    vector<mpfi_float_50> Y1;
    vector<int> index;
    vector<vector<mpfi_float_50>> result;

    // forward
    for (int i = 0; i < hidden_neural_num; i++)
    {
        mpfi_float_50 tmp = convert(0);
        for (int j = 0; j < inputs_size; j++)
        {
            tmp += I_W1[i][j] * I_X0[0][j];
        }
        tmp += I_b1[i];
        Y1.push_back(tmp);
    }

    // backward
    for (int m = 0; m < output_size; m++)
    {
        vector<mpfi_float_50> Y2;
        Y2.clear();
        for (int n = 0; n < inputs_size; n++)
        {
            mpfi_float_50 tmp = convert(0);
            for (int k = 0; k < hidden_neural_num; k++)
            {
                tmp += I_W2[m][k] * relu_d(Y1[k]) * I_W1[k][n];
            }
            Y2.push_back(tmp);
        }
        result.push_back(Y2);
    }

    mpfi_float_50 norm = convert(0);
    for (int m=0; m < inputs_size; m++) {
        norm += abs(result[Pred_X0][m]);
    }

    auto return_result = make_tuple(norm, Pred_X0);

    return return_result;
}

tuple<vector<vector<mpfi_float_50>>, mpfi_float_50> bisection(vector<vector<mpfi_float_50>> I_X0, int Pred_X0, vector<vector<long double>> W1, vector<vector<long double>> W2, vector<long double> b1, int inputs_size, int hidden_neural_num, int output_size, int count)
{

    // get bi-section list
    vector<vector<mpfi_float_50>> new_X0;

    for (int i = 0; i < I_X0.size(); i++)
    {
        vector<vector<mpfi_float_50>> tmp = getBisection(I_X0[i]);
        for (int j = 0; j < tmp.size(); j++)
        {
            new_X0.push_back(tmp[j]);
        }
    }

    // calculate interval
    vector<vector<mpfi_float_50>> wrapper;
    vector<mpfi_float_50> low;
    vector<tuple<mpfi_float_50, int>> up;
    for (int i = 0; i < new_X0.size(); i++)
    {
        wrapper.clear();
        wrapper.push_back(new_X0[i]);
        tuple<mpfi_float_50, int> tmp = myInterval(wrapper, Pred_X0, W1, W2, b1, inputs_size, hidden_neural_num, output_size);
        tuple<mpfi_float_50, mpfi_float_50> result_tmp = getLowerUpper(tmp);
        auto tumple_upper = make_tuple(get<1>(result_tmp), i);
        low.push_back(get<0>(result_tmp));
        up.push_back(tumple_upper);
    }

    // sort
    vector<mpfi_float_50> low_sorted = getSort(low);
    mpfi_float_50 low_max = low_sorted[low_sorted.size() - 1];
    // cout << "\nIn Bisection " << count << ": " << low_max << endl;

    // drop useless intervals
    vector<vector<mpfi_float_50>> result;
    for (int i = 0; i < up.size(); i++)
    {
        tuple<mpfi_float_50, int> tmp = up[i];
        mpfi_float_50 tmp_upper = get<0>(tmp);
        int index = get<1>(tmp);
        if (tmp_upper > low_max)
        {
            result.push_back(new_X0[index]);
        }
    }

    // cout << "New Size: " << result.size() << endl;

    auto result_tuple = make_tuple(result, low_max);

    return result_tuple;
}

bool conditions(int count, mpfi_float_50 low_max, bool comparedToCLEVER, int maxIterationNum, long double minGap, mpfi_float_50 up_max, int boxes_size, int max_boxes)
{
    bool result = true;
    mpfi_float_50 gap = up_max - low_max;

    // Condition 00
    if (gap < convert(minGap))
    {
        // cout << "\nExit with condition 00: reach the min gap of interval number" << endl;
        result = false;
    }

    // Condition 01
    if (comparedToCLEVER)
    {
        long double CLEVER = readParameters("parameters/CLEVER.csv")[0][0];
        if (convert(CLEVER) < low_max)
        {
            // cout << "\nExit with condition 01: low_max > CLEVER" << endl;
            result = false;
        }
    }

    // Condition 02
    if (boxes_size > max_boxes)
    {
        // cout << "\nExit with condition 02: reach max boxes number" << endl;
        result = false;
    }

    // Condition 03
    if (count >= maxIterationNum)
    {
        // cout << "\nExit with condition 03: reach max iteration number" << endl;
        result = false;
    }

    // cout << "Gap is " << gap << ". up_max is " << up_max << ". low_max is " << low_max << "." << endl;

    return result;
}

tuple<vector<vector<string>>, string> get_interval_Lipschitz_CPP(string file_index, long double radius, int inputs_size, int hidden_neural_num, int output_size, bool comparedToCLEVER, int maxIterationNum, long double minGap, int max_boxes)
{
    vector<vector<long double>> X0;
    vector<long double> tmp = {-4.832202221268014242e+00,-7.364287590384273940e+00};
    X0.push_back(tmp);
    vector<vector<long double>> W1;
    tmp = {-5.323450565338134766e-01,2.982044517993927002e-01};
    W1.push_back(tmp);
    tmp = {-4.367777407169342041e-01,1.963265985250473022e-01};
    W1.push_back(tmp);
    vector<vector<long double>> W2;
    tmp = {-3.885447978973388672e-01,4.447681903839111328e-01};
    W2.push_back(tmp);
    tmp = {-3.205857872962951660e-01,2.034754604101181030e-01};
    W2.push_back(tmp);
    vector<long double> b1;
    b1 = {-3.763356208801269531e-01,-6.647928357124328613e-01};
    int Pred_X0 = 0;
    vector<vector<mpfi_float_50>> I_X0 = expand_2D_matrix(X0, radius);

    auto start = high_resolution_clock::now();
    tuple<mpfi_float_50, int> results = myInterval(I_X0, Pred_X0, W1, W2, b1, inputs_size, hidden_neural_num, output_size);
    vector<vector<mpfi_float_50>> boxes = I_X0;
    mpfi_float_50 low_max = lower(get<0>(results)).convert_to<mpfi_float_50>();
    mpfi_float_50 up_max = upper(get<0>(results)).convert_to<mpfi_float_50>();
    int boxes_size = boxes.size();

    int count = 0;
    while (conditions(count, low_max, comparedToCLEVER, maxIterationNum, minGap, up_max, boxes_size, max_boxes))
    {
        count++;
        tuple<vector<vector<mpfi_float_50>>, mpfi_float_50> tmp = bisection(boxes, Pred_X0, W1, W2, b1, inputs_size, hidden_neural_num, output_size, count);
        boxes = get<0>(tmp);
        low_max = get<1>(tmp);
        boxes_size = boxes.size();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << " microseconds" << endl;

    vector<vector<string>> boxes_return = convert_to_string_2D_array(boxes);
    long double upper_bound = upper(get<0>(results)).convert_to<long double>();
    long double lower_bound = low_max.convert_to<long double>();
    string newResult = to_string(lower_bound) + "," + to_string(upper_bound);
    cout.precision(17);
    cout << "Interval result: " << low_max << ", " << up_max << endl;

    auto result_tuple = make_tuple(boxes_return, newResult);
    return result_tuple;
}

// PYBIND11_MODULE(IntervalCPP_ReLU_V2, m)
// {
//     m.def("get_interval_Lipschitz_CPP", &get_interval_Lipschitz_CPP);
// }

int main() {
    string file_index="0";
    long double radius = 1e-7;
    int inputs_size = 2;
    int hidden_neural_num = 2; 
    int output_size = 2;
    bool comparedToCLEVER = false; 
    int maxIterationNum = 10;
    long double minGap = 1e-6;
    int max_boxes = 10000;
    
    get_interval_Lipschitz_CPP(file_index, radius, inputs_size, hidden_neural_num, output_size, comparedToCLEVER, maxIterationNum, minGap, max_boxes);
    
    return 0;
}