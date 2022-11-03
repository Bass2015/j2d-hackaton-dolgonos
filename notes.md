`n_estimators`: number of trees
- not necesary to tune. Large number is ok

`criterion`: gini | entropy 
- Try both
  
`max_depth`
- No so necessary, pick reasonable value
  
`max_features`
- Small value -> reduce variance, high individual tree bias
- High value -> decrease bias
- If data clean -> number of random features can be small
- If noisy data -> better higher value to increase chances of quality feature being included
- sqrt normally good for classification
- Try various decimal values too.
  
`max_samples`
- This is the bootstrap size
- Normally the best thing is to take all the dataset
