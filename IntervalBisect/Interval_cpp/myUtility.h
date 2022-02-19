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
#include <cstdlib>

using namespace std;
using namespace boost;
using namespace boost::multiprecision;

mpfi_float_50 expand(long double x, long double r);

vector<mpfi_float_50> expand_array(vector<long double> arr, long double r);

vector<vector<mpfi_float_50>> expand_2D_matrix(vector<vector<long double>> matrix, long double r);

// vector<vector<mpfi_float_50>> expand_3D_matrix(vector<vector<vector<long double>>> matrix, long double r);

mpfi_float_50 convert(long double x);

vector<mpfi_float_50> convert_array(vector<long double> arr);

vector<vector<mpfi_float_50>> convert_2D_matrix(vector<vector<long double>> matrix);

mpfi_float_50 relu(mpfi_float_50 x);

mpfi_float_50 relu_d(mpfi_float_50 x);

mpfi_float_50 sigmoid(mpfi_float_50 x);

mpfi_float_50 sigmoid_d(mpfi_float_50 x);

tuple<mpfi_float_50, int> myMax(vector<mpfi_float_50> arr);

vector<vector<mpfi_float_50>> repeatVector(vector<vector<mpfi_float_50>> X0);

vector<vector<mpfi_float_50>> getBisection(vector<mpfi_float_50> I_X0);

vector<vector<mpfi_float_50>> getBisectionInfluence(vector<mpfi_float_50> I_X0, int influenceIndex);

void print2DIntervalVector(vector<vector<mpfi_float_50>> X);

void print2DFloatVector(vector<vector<long double>> X);

void print2DStringVector(vector<vector<string>> X);

vector<vector<long double>> readParameters(string filename);

vector<vector<vector<long double>>> readParametersMulti(string filename);

tuple<mpfi_float_50, mpfi_float_50> getLowerUpper(tuple<mpfi_float_50, int> X);

tuple<mpfi_float_50, mpfi_float_50> getLowerUpperIA(tuple<mpfi_float_50, int, vector<int>> X);

vector<mpfi_float_50> getSort(vector<mpfi_float_50> X);

vector<tuple<mpfi_float_50, int>> getSort_tuple(vector<tuple<mpfi_float_50, int>> X);

vector<vector<string>> convert_to_string_2D_array(vector<vector<mpfi_float_50>> array);

vector<vector<string>> convert_to_string_2D_array_IA(vector<tuple<vector<mpfi_float_50>, vector<int>>> array);

bool compareIntervalGreater(mpfi_float_50 x1, mpfi_float_50 x2);

mpfi_float_50 abs_d(mpfi_float_50 x);