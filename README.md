# plot_ml_metric
plot machine learning metrics, such as regression and classification

## test biclassify metrics
```
from plot_ml_metric import bi_classify_metrics
labels = np.array([[1,0], [1,0], [1,0], [0,1], [1,0],
                   [0,1], [0,1], [0,1], [0,1], [0,1],])
preds = np.array([[0.9, 0.1], [0.17, 0.83], [0.83, 0.17], [0.19, 0.81], [0.70, 0.30], 
                  [0.35, 0.65], [0.99, 0.01], [0.25, 0.75], [0.79, 0.21], [0.74, 0.26],])
bi_classify_metrics(labels, preds, plot_cm=True, plot_auc=True, save_path_name='bi',classnames=['Negative','Positive'])
```
## test mulitclassify metrics
```
from plot_ml_metric import multi_classify_metrics
labels = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0],
                    [0, 1, 0], [0,  1, 0], [0, 1, 0],
                    [0, 0, 1], [0, 0, 1], [0, 0, 1],])
preds = np.array([[0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
                [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
                [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],])
multi_classify_metrics(labels, preds, 'micro', plot_auc=True, plot_cm=True, save_path_name="multi",classnames=['a','b','c'])
```

## test regress metrics
```
from plot_ml_metric import regression_metrics
actual_values = np.array([3, -0.5, 2, 7, 4.2])
predicted_values = np.array([2.5, 0.0, 2, 8, 4.1])
regression_metrics(actual_values, predicted_values, plot_regress=True, save_path_name='regress')
```

## test mulit regress metrics
```
from plot_ml_metric import plot_multiple_regressions
actual_values1 = np.array([3, -0.5, 2, 7, 4.2])
predicted_values1 = np.array([2.5, 3, 2, 8, 4.1])
actual_values2 = np.array([1, 2, 3, 4, 5])
predicted_values2 = np.array([1.1, 2.2, 3.1, 3.9, 5.0])
plot_multiple_regressions([actual_values1, actual_values2], [predicted_values1, predicted_values2], ['model1', 'model2'], 'multiple_regression_plot.png')
```
