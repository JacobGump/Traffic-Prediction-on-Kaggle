# Traffic-Prediction-on-Kaggle
competition on Kaggle - "traffic speed prediction in 2019 fall semester at Peking University"

### Task
Consider the traffic speed prediction problem for 228 sensor stations in a region. One day is splitted into 288 time periods (5 minutes per time period), and an average speed in each time period for every sensor station is counted.

The problem presented is: __Given the speed of the previous hour (12 time periods) of these 228 stations and to predict the speed after t=15, 30, 45 minutes.__

Note that the __distance__ relationship of these 228 sensor stations is given as well, we may be able to consider the interaction of their traffic conditions through these distance information.

### Solution
We have applied three established methods onto the task of traffic prediction. Namely we first used an __Auto-Encoder__ model to map the input data to the prediction. Then we added in the distance information by using a __Graph Convolutional Network__. Finally, we did a experiment using __Spatio-Temporal Graph Convolutional Network__, which explicitly encodes the spatiol and temporal information of the data.

The codes and results for all three experiments is included in this repository.

### Features
1. We achieved first place in Leaderboard for the task on Kaggle(1st among 27)
2. We designed and implemented an auto-encoder model in tensorflow
3. We implemented a dataloader and adjusted the code of GCN to apply onto the task

### Result
| Model     | RMSE     |
| --------- | -------- |
| AE        | 13.66    |
| GCN       | 7.92     |
| __STGCN__ | __4.82__ |
