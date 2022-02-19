#include <boost/multiprecision/mpfi.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/tokenizer.hpp>
#include <boost/regex.hpp>
#include <math.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace boost;
using namespace boost::multiprecision;

mpfi_float_50 expand(float x, float r);

vector<mpfi_float_50> expand_array(vector<float> arr, float r);

vector<vector<mpfi_float_50>> expand_2D_matrix(vector<vector<float>> matrix, float r);

// vector<vector<mpfi_float_50>> expand_3D_matrix(vector<vector<vector<float>>> matrix, float r);

mpfi_float_50 convert(float x);

vector<mpfi_float_50> convert_array(vector<float> arr);

vector<vector<mpfi_float_50>> convert_2D_matrix(vector<vector<float>> matrix);

mpfi_float_50 relu(mpfi_float_50 x);

mpfi_float_50 relu_d(mpfi_float_50 x);

mpfi_float_50 sigmoid(mpfi_float_50 x);

tuple<mpfi_float_50, int> myMax(vector<mpfi_float_50> arr);

vector<vector<mpfi_float_50>> repeatVector(vector<vector<mpfi_float_50>> X0);

vector<vector<mpfi_float_50>> getBisection(vector<mpfi_float_50> I_X0);

vector<vector<mpfi_float_50>> getBisectionInfluence(vector<mpfi_float_50> I_X0, int influenceIndex);

void print2DIntervalVector(vector<vector<mpfi_float_50>> X);

void print2DFloatVector(vector<vector<float>> X);

void print2DStringVector(vector<vector<string>> X);

vector<vector<float>> readParameters(string filename);

vector<vector<vector<float>>> readParametersMulti(string filename);

tuple<mpfi_float_50, mpfi_float_50> getLowerUpper(tuple<mpfi_float_50, int> X);

vector<mpfi_float_50> getSort(vector<mpfi_float_50> X);

vector<tuple<mpfi_float_50, int>> getSort_tuple(vector<tuple<mpfi_float_50, int>> X);

vector<vector<string>> convert_to_string_2D_array(vector<vector<mpfi_float_50>> array);

bool compareIntervalGreater(mpfi_float_50 x1, mpfi_float_50 x2);