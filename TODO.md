# INVESTAI

-------------------- -------------------- --------------------
-------------------- -------------------- --------------------

## TODO
-
- - jak predikovat navratnost vzhledem k propadu
- - jak zajistit aktualizaci portfolia?
- - jak zajistit, kdy je lepsi drzet cash?
- - jak trenovat 500 akcii ale vybrat potom pouze 20?
- - zkusit ruzne hyperparameters
- - zkusit ruzne hyperparameters

- Create Data Sets
  - Evaluate 2 FinRL-Meta examples
  - 1st Experiment - Make data set bigger
    - Create Data Set
    - Create Environment
    - Evaluate (Train + Test)
  - 2st Experiment - Add more Ratios into original data set
    - Create Data Set
    - Create Environment
    - Evaluate (Train + Test)
  - 3st Experiment - Make bigger data set + Add more Ratios into original data set
    - Create Data Set
    - Create Environment
    - Evaluate (Train + Test)
  - 4th Experiment - Averaged Price Data along the time
    - Create Data Set
    - Create Environment
    - Evaluate (Train + Test)

-------------------- -------------------- --------------------
-------------------- -------------------- --------------------

## BASICS

- Update forked finrl-meta to latest version
- Clean up the code
- Add more comments
- Add more tests
- Add more documentation
- Add more examples

### IDEAS

- Build (Neural network | ...)
- which will (train NN | ...)
- that can predict the (next move | long prediction | ...)
- based on the (change of any fundamental data | sentiment).
- RL train multiple agents
- Binary classifier that subject to trade will go up/down (SVM)
- Finite/infinite automatically building state machine consists in probabilities
- Find correlation between some features (indicators | ...) and result ±1-5% (Gaussian Distribution)

#### DIVERSIFICATION

- Using:
  - Investing into different sectors

#### RL: Features

- Which companies have the most amount of cash available relative to amount of debt?
- Which companies have the biggest profit relative to amount of debt?
- Which companies have the biggest sales margin?
- Which companies have the biggest sales margin?
- Which companies are increasing profits constantly
- Which companies are decreasing debts constantly?
- Which companies are increasing dividend yield?
- Which companies are the oldest?
- Which companies are the youngest?

##### HINTS

- The company exist a long time so the growth can be slower than in the young company
- Result: (make the union on the top most companies | train NN | ...)
- combination of financial series classification and portfolio optimization surpasses each of the single approaches

-------------------- -------------------- --------------------
