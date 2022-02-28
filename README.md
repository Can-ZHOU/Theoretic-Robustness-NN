# A Domain-Theoretic Framework for Robustness Analysis of Neural Network - lipDT

## Code Structure
```
lipDT
|
|--- data
|   |
|   |--- MNIST
|   |--- Iris_modified.csv
|
|--- IntervalBisect
|   |
|   |--- Interval_cpp
|   |--- parameters
|   |--- intervalBisect.py
|   |--- IntervalCPP_ReLU_V2.cpython-38-x86_64-linux-gnu.so
|   |--- MNIST01
|   |--- MultiLayers
|   |--- utilities.py
|
|--- OtherMethods
|   |
|   |--- lipMIP
|   |--- other_methods (clever, fast_lip, seq_lip, naive_methods)
|   |--- ZLip
|   |--- CLEVER.py
|
|--- parameters
|
|--- Saved 
|   |--- AccuaryEfficiency
|   |--- convergency
|   |--- IA
|   |--- MultiLayers
|   |--- MyRandom
|   |--- WeightBias
|
|--- accuaryEfficiency.py (run this file for accuary & efficiency experiemnts)
|
|--- convergency.py (run this file for convergency experiemnts)
|
|--- experiments.py
|--- IA.py
|
|--- multiLayers.py (run this file for multi-layers experiments)
|
|--- RE.py (run this to get relative errors in the paper table)
|
|--- README.md
|
|--- utilities.py
|
|--- weightBias.py (run this file for weight & bias experiemnts)
```

## Experiments
### Accuary & Efficiency
#### Code file: 
- accuaryEfficiency.py

#### Results:
- Random01
```
interval Cpp:
8289 microseconds
Interval result: 0.322705690859059757, 0.322705690859059757
interval ReLU: value - interval([0.32270569085905976]) timing - 338358000
lipMIP ReLU: value - 0.041121020913124084 timing - 10034800
ZLip ReLU: value - 0.041121020913124084 timing - 2507500
CLEVER ReLU: value - 0.2815846800804138 timing - 1054102200
FastLip ReLU: value - 0.041121020913124084 timing - 845100
NaiveUB ReLU: value - 0.6921076774597168 timing - 95900
RandomLB ReLU: value - 0.2815846800804138 timing - 137321600
SeqLip ReLU: value - 0.2129872590303421 timing - 1963200
```
- Random02
```
interval Cpp:
318 microseconds
Interval result: 0.571150659783544512, 0.614485302898467367
interval ReLU: value - interval([0.5711506597835445, 0.6144853028984674]) timing - 12778100
lipMIP ReLU: value - 0.043334643114922855 timing - 9840500
ZLip ReLU: value - 0.04333464428782463 timing - 2625300
CLEVER ReLU: value - 0.04333464428782463 timing - 1102013000
FastLip ReLU: value - 0.04333464428782463 timing - 623500
NaiveUB ReLU: value - 0.7291175127029419 timing - 51100
RandomLB ReLU: value - 0.04333464428782463 timing - 143504000
SeqLip ReLU: value - 0.4503255784511566 timing - 731600
```
- Random03
```
interval Cpp:
12516 microseconds
Interval result: 0.614485302898467367, 0.614485302898467367
interval ReLU: value - interval([0.6144853028984674]) timing - 543024100
lipMIP ReLU: value - 0.043334643114922855 timing - 3105500
ZLip ReLU: value - 0.04333464428782463 timing - 1516900
CLEVER ReLU: value - 0.04333464428782463 timing - 1105068400
FastLip ReLU: value - 0.04333464428782463 timing - 754100
NaiveUB ReLU: value - 0.7291175127029419 timing - 65100
RandomLB ReLU: value - 0.04333464428782463 timing - 144733500
SeqLip ReLU: value - 0.4503255784511566 timing - 565400
```
- IRIS
```
interval Cpp:
59 microseconds
Interval result: 3.48780926173324346, 3.48780926173324346
interval ReLU: value - interval([3.4878092617332435]) timing - 1582300
lipMIP ReLU: value - 3.4878092408180237 timing - 2395500
ZLip ReLU: value - 3.487809181213379 timing - 1100000
CLEVER ReLU: value - 3.487809181213379 timing - 1129720600
FastLip ReLU: value - 3.487809181213379 timing - 626700
NaiveUB ReLU: value - 12.569558143615723 timing - 50000
RandomLB ReLU: value - 3.487809181213379 timing - 139967700
SeqLip ReLU: value - 1.8214526176452637 timing - 749400
```
- MNIST
```
Using MNIST dataset.
interval Cpp:
Interval result: 0.34815343269413546, 0.34815343269413546
164040 microseconds
0
interval ReLU: value - interval([0.34815343269412385, 0.3481534326941471]) timing - 2388625100
lipMIP ReLU: value - 0.3481534055299562 timing - 33178800
ZLip ReLU: value - 0.34815341234207153 timing - 7189500
CLEVER ReLU: value - 0.3481535315513611 timing - 4308822000
FastLip ReLU: value - 0.34815341234207153 timing - 884400
NaiveUB ReLU: value - 107.7006607055664 timing - 75900
RandomLB ReLU: value - 0.3481535315513611 timing - 177969900
SeqLip ReLU: value - 0.670467323694055 timing - 4930500
```

#### Saved Information
- folder: Saved/AccuaryEfficiency

### Weight & Bias
#### Code file:
- weightBias.py
#### Saved Information
- folder: Saved/WeightBias

### Convergency
#### Code file:
- convergency.py
#### Saved Information
- folder: Saved/Convergency

### MultiLayers
#### Code file:
- multiLayers.py
#### Saved Information
- folder: Saved/MultiLayers

### Influence Analysis
- IntervalBisect/Interval_cpp/test_sigmoid01

## Environment & Dependency & Other information
### EnvironmentL
- Ubuntu 20.04 LTS
### Dependency (only for lipDT)
- ART (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- MPFI (boost library in cpp)
- matplotlib 3.3.3
- numpy 1.19.4
- pybind11 2.6.1
- pyinterval 1.2.1
- Python 3.8.5
- pytorch 1.7.0
### Other information
- Other methods:
    - lipMIP & Zlip: https://github.com/revbucket/lipMIP
    - SeqLip & fastLip: https://github.com/revbucket/lipMIP/tree/master/other_methods
    - Clever: https://github.com/Trusted-AI/adversarial-robustness-toolbox
- All codes from other methods may be modified and used in lipDT.