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
#include "myUtilityMNIST.h"

using namespace std;
using namespace std::chrono;
using namespace boost::multiprecision;

tuple<mpfi_float_50, int> myInterval(vector<mpfi_float_50> I_X, int outputIndex, vector<int> layer_sizes, vector<vector<vector<float>>> weights, vector<vector<float>> bias)
{
    // init
    vector<vector<mpfi_float_50>> result;
    vector<vector<vector<mpfi_float_50>>> gradient;

    int hiddenLayerNumber = layer_sizes.size()-2;
    int inputs_size = layer_sizes[0];

    for (int k = 0; k < hiddenLayerNumber; k++){
        vector<mpfi_float_50> tmpResult;
        for (int i = 0; i < layer_sizes[k+1]; i++) {
            mpfi_float_50 tmp = convert(0);
            for (int j = 0; j < layer_sizes[k]; j++) {
                if (k == 0)
                    tmp += weights[k][i][j] * I_X[j];
                else
                    tmp += weights[k][i][j] * relu(result[k-1][j]);
            }
            tmp += bias[k][i];
            tmpResult.push_back(tmp);
        }
        result.push_back(tmpResult);
    }

    for (int k = 1; k < hiddenLayerNumber+1; k++) {
        vector<vector<mpfi_float_50>> gradientOutput;
        for (int outputIndex = 0; outputIndex < layer_sizes[k+1]; outputIndex++) {
            vector<mpfi_float_50> gradientInput;
            for (int inputIndex = 0; inputIndex < inputs_size; inputIndex++) {
                mpfi_float_50 gradientTmp = convert(0);
                if (k == 1) {
                    for (int i = 0; i < layer_sizes[k]; i++) {
                        gradientTmp += weights[k][outputIndex][i] * relu_d(result[k-1][i]) * weights[k-1][i][inputIndex];
                    }
                } else {
                    for (int i = 0; i < layer_sizes[k]; i++) {
                        gradientTmp += weights[k][outputIndex][i] * relu_d(result[k-1][i]) * gradient[k-2][i][inputIndex];
                    }
                }
                gradientInput.push_back(gradientTmp);
            }
            gradientOutput.push_back(gradientInput);
        }
        gradient.push_back(gradientOutput);
    }

    mpfi_float_50 norm = convert(0);
    for (int m=0; m < inputs_size; m++) {
        norm += abs(gradient[hiddenLayerNumber-1][outputIndex][m]);
    }

    auto return_result = make_tuple(norm, outputIndex);

    return return_result;
}

tuple<vector<vector<mpfi_float_50>>, mpfi_float_50> bisection(vector<vector<mpfi_float_50>> I_X0, int outputIndex, vector<int> layer_sizes, vector<vector<vector<float>>> weights, vector<vector<float>> bias, int count)
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
    vector<mpfi_float_50> low;
    vector<tuple<mpfi_float_50, int>> up;
    for (int i = 0; i < new_X0.size(); i++)
    {
        tuple<mpfi_float_50, int> tmp = myInterval(new_X0[i], outputIndex, layer_sizes, weights, bias);
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

bool conditions(int count, mpfi_float_50 low_max, int maxIterationNum, float minGap, mpfi_float_50 up_max, int boxes_size, int max_boxes)
{
    bool result = true;
    mpfi_float_50 gap = up_max - low_max;

    // Condition 00
    if (gap < convert(minGap))
    {
        // cout << "\nExit with condition 00: reach the min gap of interval number" << endl;
        result = false;
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

tuple<vector<vector<string>>, string> get_interval_Lipschitz_CPP(string file_index, vector<int> layer_sizes, vector<vector<mpfi_float_50>> I_X0, int maxIterationNum, float minGap, int max_boxes, vector<vector<vector<float>>> weights, vector<vector<float>> bias)
{
    int outputIndex = 0;
    tuple<mpfi_float_50, int> results = myInterval(I_X0[0], outputIndex, layer_sizes, weights, bias);

    vector<vector<mpfi_float_50>> boxes = I_X0;
    mpfi_float_50 low_max = lower(get<0>(results)).convert_to<mpfi_float_50>();
    mpfi_float_50 up_max = upper(get<0>(results)).convert_to<mpfi_float_50>();
    int boxes_size = boxes.size();

    int count = 0;
    while (conditions(count, low_max, maxIterationNum, minGap, up_max, boxes_size, max_boxes))
    {
        count++;
        tuple<vector<vector<mpfi_float_50>>, mpfi_float_50> tmp = bisection(I_X0, outputIndex, layer_sizes, weights, bias, count);
        boxes = get<0>(tmp);
        low_max = get<1>(tmp);
        boxes_size = boxes.size();
    }

    vector<vector<string>> boxes_return = convert_to_string_2D_array(boxes);

    float upper_bound = upper(get<0>(results)).convert_to<float>();
    float lower_bound = low_max.convert_to<float>();
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
    float radius = 0.001;
    string file_index = "7";
    string path = "parameters/E" + file_index;
    int maxIterationNum = 0;
    float minGap = 1e-6;
    int max_boxes = 3000;
    vector<int> layer_sizes = {784, 10, 10, 2};
    vector<vector<float>> X0 = readParameters(path + "/X0.csv");
    vector<vector<mpfi_float_50>> I_X0 = expand_2D_matrix(X0, radius);
    vector<vector<vector<float>>> weights = readParametersMulti(path + "/weights.csv");
    vector<vector<float>> bias = readParameters(path + "/bias.csv");
    auto start = high_resolution_clock::now();
    get_interval_Lipschitz_CPP(file_index, layer_sizes, I_X0, maxIterationNum, minGap, max_boxes, weights, bias);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << " microseconds" << endl;
    return 0;
}