import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


images_dir = "Images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


data = pd.read_csv('ENData.csv')


data = data.drop(['ID'], axis=1)


target_vars = ['Intelligibility', 'IntelligibilityTransformer', 'Comprehensibility']


evaluation_metrics = pd.DataFrame()


all_feature_coef = pd.Series()

for target in target_vars:
    
    X = data.drop(target_vars, axis=1)
    y = data[target]
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3407)
    
    
    lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000000)
    lasso_cv.fit(X_train, y_train)
    
    lasso = Lasso(alpha=lasso_cv.alpha_)
    lasso.fit(X_train, y_train)
    
    y_pred_all = lasso_cv.predict(X_scaled)
    y_pred_train = lasso_cv.predict(X_train)
    y_pred_test = lasso_cv.predict(X_test)
    
    
    evaluation_metrics[f'{target}-Alpha'] = [lasso_cv.alpha_]
    
    evaluation_metrics[f'{target}-Train R^2'] = [r2_score(y_train, y_pred_train)]
    evaluation_metrics[f'{target}-Test R^2'] = [r2_score(y_test, y_pred_test)]
    evaluation_metrics[f'{target}-All R^2'] = [r2_score(y, y_pred_all)]
    evaluation_metrics[f'{target}-Train MSE'] = [mean_squared_error(y_train, y_pred_train)]
    evaluation_metrics[f'{target}-Test MSE'] = [mean_squared_error(y_test, y_pred_test)]
    evaluation_metrics[f'{target}-All MSE'] = [mean_squared_error(y, y_pred_all)]
    evaluation_metrics[f'{target}-Train RMSE'] = [np.sqrt(mean_squared_error(y_train, y_pred_train))]
    evaluation_metrics[f'{target}-Test RMSE'] = [np.sqrt(mean_squared_error(y_test, y_pred_test))]
    evaluation_metrics[f'{target}-All RMSE'] = [np.sqrt(mean_squared_error(y, y_pred_all))]
    
    
    feature_coef = pd.DataFrame({
        'Feature': X.columns,
        'Coefficients (B)': lasso.coef_
    })
    
    feature_coef.to_csv('./CoefficientsEN/'+target+'FeaturesCoefficients.csv', header=True)
    
    
    max_length = feature_coef['Feature'].str.len().max()

    
    feature_coef['Feature'] = feature_coef['Feature'].apply(lambda x: x.ljust(max_length))

    
    feature_coef['Representative Name'] = feature_coef.apply(lambda x: ' ' if x['Coefficients (B)'] == 0 else x['Feature'], axis=1)
    
    
    top_features_abs_sorted = feature_coef['Coefficients (B)'].abs().sort_values(ascending=False).head(10)
    top_features = feature_coef.loc[top_features_abs_sorted.index]  

    
    top_features_sorted = top_features.sort_values(by='Coefficients (B)')

    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    
    fig, ax = plt.subplots(figsize=(10, 8))

    
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = plt.Normalize(-0.05, .05)
    cmap = plt.cm.coolwarm

    
    colors = cmap(norm(top_features_sorted['Coefficients (B)'].values))

    
    for i, (coef, name) in enumerate(zip(top_features_sorted['Coefficients (B)'], top_features_sorted['Representative Name'])):
        ax.barh(y=i, width=coef, color=colors[i])

    
    ax.set_yticks(range(len(top_features_sorted)))
    ax.set_yticklabels(top_features_sorted['Representative Name'])

    
    plt.xlabel('Coefficients', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    if target == 'IntelligibilityTransformer':
        plt.xlim(-0.15, 0.15)
    elif target == 'Intelligibility':
        plt.xlim(-0.05, 0.05)
    elif target == 'Comprehensibility':
        plt.xlim(-0.09, 0.09)
    plt.title(f'Top Features Selected by Lasso for {y.name} in EN', fontsize=16)
    plt.tight_layout()

    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Coefficient Value')

    
    plt.savefig(f'{images_dir}/FeatureCoefficients{y.name}EN.jpg', format='jpg')
    plt.close()




evaluation_metrics = evaluation_metrics.T
evaluation_metrics.columns = ['Value']
evaluation_metrics.to_csv('ModelEvaluationMetricsEN.csv')
