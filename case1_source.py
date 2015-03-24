# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:54:08 2015

@author: bordingj
"""
import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def transform_data(X, feature_importance, num_features):
    return X[:,np.argsort(feature_importance)[-num_features:]]

def plot_CV_GRID_and_getOptimum(CV_grid_array, x_linspace, y_linspace,
                                x_name, y_name, title):
    import matplotlib
    import matplotlib.pyplot as plt
    
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    X, Y = np.meshgrid(x_linspace, y_linspace)
    Z = CV_grid_array.T
    plt.figure(figsize=(13,8))
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.colorbar(shrink=0.5, aspect=5, label='RMSE')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    best_ind = np.unravel_index(np.argmin(CV_grid_array),CV_grid_array.shape)
    opt_x=  x_linspace[best_ind[0]]
    opt_y = y_linspace[best_ind[1]]
    plt.plot([opt_x], [opt_y], 'g.', markersize=20.0)
    best_RMSE =CV_grid_array[best_ind[0],best_ind[1]]
    return (best_RMSE, opt_x, opt_y)

def AddWeatherData(df_in):
    df = df_in.copy()
    # Import weather data
    
    weather_data = pd.read_csv('/home/bordingj/Copy/Courses/Computational_DataAnalysis/Case_1/weatherData.csv')
    # convert dates to datetime format
    weather_data['DATE'] = pd.to_datetime(weather_data['DATE'],format="%Y%m%d")
    weather_data.set_index(['DATE'], inplace=True)
    # discard variables which obviously can not be predictors
    weather_data.drop(['STATION','PGTM','FMTM'], axis=1, inplace=True)
    # get JFK weather only
    weather_data = weather_data[weather_data['STATION_NAME'] == 'NEW YORK J F KENNEDY INTERNATIONAL AIRPORT NY US']
    weather_data.drop('STATION_NAME', axis=1,inplace=True)
    """
    # lets see how much missing data each column has (missing values are encoded as -9999 and 9999)

    for (col_num, col) in enumerate(weather_data.columns):
    print('Column {0} (column no. {1}) has {2} % missing'.format( col, col_num, 
          100*np.sum(weather_data[col].isin([-9999,9999]))/weather_data.shape[0])
          )
    """
    # We descard all columns from column 8 to end because they have mostly missing data and set other missing data as NaN.
    weather_data = weather_data.iloc[:,:8]
    for colname in weather_data.columns:
        weather_data.loc[weather_data[colname].isin([-9999,9999]),colname] = np.nan
    
    # Next, we linearly interpolate remaining missing data with respect to time
    
    for colname in weather_data.columns:
        weather_data[colname].interpolate(method="time",inplace=True)
        
    df['day'] = df.index.date
    weather_data.index = weather_data.index.date
    df = df.join(weather_data, on='day', how='left')
    df.drop('day', axis=1, inplace=True)
    return df

def Generate_df_with_Dummies(df_in, remove_infrequencies=True):
    df = df_in.copy()
    df = pd.concat([df, pd.get_dummies(df['TailNum'], prefix='TailNum')], axis=1)
    df.drop('TailNum', axis=1, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Dest'], prefix='Dest')], axis=1)
    df.drop('Dest', axis=1, inplace=True)
    # Exclude tail-numbers / destinations with less than 20 occorences in data set
    if remove_infrequencies:
        bool_series = (df.iloc[:,df_in.shape[1]:].sum() >= 20)
        df = df[df.columns[:df_in.shape[1]].tolist()+bool_series[bool_series].index.tolist()]
    return df
        
class Case1(object):
    def __init__(self):
        self.df = 0
        
    def Wrangle_TrainingData(self):
        #read data
        try:
            df = pd.read_pickle('FlightDataTraining')
        except:
            df = pd.read_excel('FlightDataTraining.xlsx')
            df.to_pickle('FlightDataTraining')
        #Setting time as index and sort by time
        df['datetime']=""
        for i in range(df.shape[0]):
            timestamp = df['CRSDepTime'][i]/100
            hour = math.floor(timestamp)
            minute = int(round((timestamp-hour)*100))
            df['datetime'][i] = pd.datetime(df['Year'][i], 
                                            df['Month'][i], df['DayofMonth'][i],
                                            hour, minute)
        df.set_index(['datetime'], inplace=True)
        df.sort_index(ascending=False, inplace=True)
        
        #add Departures per day variable
        Dep_per_day = pd.DataFrame(pd.Series(df.index.date).value_counts(), columns=['Dep_per_day'])
        df['day'] = df.index.date
        df = df.join(Dep_per_day, on='day', how='left')
        df.drop('day', axis=1, inplace=True)
        
        #Remove rows with nans in departure delay
        df = df.loc[~np.isnan(df['DepDelay']),:]
        
        #log-transform Data
        logtransform_Constant = np.max(np.abs(df['DepDelay']))+1
    
        df['LogDepDelay'] = np.log(df['DepDelay']+logtransform_Constant)
        
        #'UniqueCarrier' and 'Origin' are redundant. 
        # Moreoever, we dont expect 'FlightNum' to have any influence on the delay. 
        # So lets remove these columns.
        df.drop(['FlightNum', 'Origin', 'UniqueCarrier'], axis=1, inplace=True)
        
        # We speculate that the time difference between each departure might have some effect on delays,  
        # so lets calculate that and add it as a new feature
        df['tvalue'] = df.index
        delta = ((df.ix[:-1,'tvalue'].as_matrix()-df.ix[1:,'tvalue'].as_matrix())/(1e9*60)).astype(int)
        df['deltaTime'] = np.append(delta, 0)
        df.drop('tvalue', axis=1, inplace=True)
        self.df = df
        self.logtransform_Constant = logtransform_Constant

    def GenerateCVindexes(self, n_folds):
        from sklearn import cross_validation
        self.n_folds = n_folds
        return cross_validation.KFold(self.df.shape[0], n_folds=n_folds,  shuffle=True)
    
    def GenerateFeatureImportance_CV_lists(self, kf):
        df_baseline = Generate_df_with_Dummies(self.df)
        len_of_dummies = df_baseline.shape[1] - (self.df.shape[1] - 2)
        y = df_baseline.pop('DepDelay').as_matrix()
        y_log = df_baseline.pop('LogDepDelay').as_matrix()
        X = df_baseline.as_matrix()
        d = X.shape[1]-len_of_dummies
        from sklearn.ensemble import ExtraTreesRegressor
        feature_importance_CV_list = []
        for train_index, val_index in kf:
            # Split data
            X_train= X[train_index].copy()
            y_train_log= y_log[train_index].copy()
            # Generate Z scores parameters
            Scaler = StandardScaler()
            Scaler.fit(X_train[:,:d])
            X_train[:,:d] = Scaler.transform(X_train[:,:d])
            ######## Feature Selection #############
            Regr = ExtraTreesRegressor(n_estimators=100, n_jobs=4)
            Regr.fit(X_train, y_train_log)
            feature_importance_CV_list.append(Regr.feature_importances_)
        return feature_importance_CV_list
    
    def CrossValidation(self, Regr, kf, use_y_log = True, standardization=False, feature_importance_list=None, 
                        num_features=None):
        
        df_baseline = Generate_df_with_Dummies(self.df)
        len_of_dummies = df_baseline.shape[1] - (self.df.shape[1] - 2)
        y = df_baseline.pop('DepDelay').as_matrix()
        y_log = df_baseline.pop('LogDepDelay').as_matrix()
        X = df_baseline.as_matrix()
        d = X.shape[1]-len_of_dummies
        if num_features is None:
            num_features = X.shape[1]
        mse_val_mean = 0
        for k,(train_index, val_index) in enumerate(kf):
            # Split data
            X_train, X_val = X[train_index].copy(), X[val_index].copy()
            y_train, y_train_log, y_val = y[train_index], y_log[train_index], y[val_index]
            if standardization:
                Scaler = StandardScaler()
                Scaler.fit(X_train[:,:d])
                X_train[:,:d] = Scaler.transform(X_train[:,:d])
                X_val[:,:d] = Scaler.transform(X_val[:,:d])
            ######## Feature Selection #############
            if feature_importance_list is not None:
                feature_importance = feature_importance_list[k]
                X_train = transform_data(X_train, feature_importance, num_features)
                X_val = transform_data(X_val, feature_importance, num_features)
            ####### Model training ##########
            if use_y_log:
                y_pred_log = Regr.fit(X_train, y_train_log).predict(X_val)
                y_pred = np.exp(y_pred_log) -self.logtransform_Constant
            else:
                y_pred = Regr.fit(X_train, y_train).predict(X_val)

            mse_val_mean += mean_squared_error(y_val,y_pred)
        mse_val_mean = mse_val_mean/self.n_folds
        return np.sqrt(mse_val_mean)
    
    def Wrangle_EvalData(self):
        #read data
        try:
            df = pd.read_pickle('FlightDataEvalInput')
        except:
            df = pd.read_excel('FlightDataEvalInput.xlsx')
            df.to_pickle('FlightDataEvalInput')
        #Setting time as index
        df['datetime']=""
        for i in range(df.shape[0]):
            timestamp = df['CRSDepTime'][i]/100
            hour = math.floor(timestamp)
            minute = int(round((timestamp-hour)*100))
            df['datetime'][i] = pd.datetime(df['Year'][i], 
                                            df['Month'][i], df['DayofMonth'][i],
                                            hour, minute)
        df['Original index'] = df.index
        df.set_index(['datetime'], inplace=True)
        
        df.sort_index(ascending=False, inplace=True)
        
        #add Departures per day variable
        Dep_per_day = pd.DataFrame(pd.Series(df.index.date).value_counts(), columns=['Dep_per_day'])
        df['day'] = df.index.date
        df = df.join(Dep_per_day, on='day', how='left')
        df.drop('day', axis=1, inplace=True)
        
        #'UniqueCarrier' and 'Origin' are redundant. 
        # Moreoever, we dont expect 'FlightNum' to have any influence on the delay. 
        # So lets remove these columns.
        df.drop(['FlightNum', 'Origin', 'UniqueCarrier', 'DepDelay'], axis=1, inplace=True)
        
        # We speculate that the time difference between each departure might have some effect on delays,  
        # so lets calculate that and add it as a new feature
        df['tvalue'] = df.index
        delta = ((df.ix[:-1,'tvalue'].as_matrix()-df.ix[1:,'tvalue'].as_matrix())/(1e9*60)).astype(int)
        df['deltaTime'] = np.append(delta, 0)
        df.drop('tvalue', axis=1, inplace=True)
        self.df_eval = df
    
    def EstimateRMSEforMyEnsemble(self, Regr, kf, standardization=False, feature_importance_list=None, 
                        num_features=None):
        
        df_baseline = Generate_df_with_Dummies(self.df)
        len_of_dummies = df_baseline.shape[1] - (self.df.shape[1] - 2)
        y = df_baseline.pop('DepDelay').as_matrix()
        y_log = df_baseline.pop('LogDepDelay').as_matrix()
        X = df_baseline.as_matrix()
        d = X.shape[1]-len_of_dummies
        mse_val_mean = 0
        for k,(train_index, val_index) in enumerate(kf):
            # Split data
            X_train, X_val = X[train_index].copy(), X[val_index].copy()
            y_train, y_train_log, y_val = y[train_index], y_log[train_index], y[val_index]
            if standardization:
                Scaler = StandardScaler()
                Scaler.fit(X_train[:,:d])
                X_train[:,:d] = Scaler.transform(X_train[:,:d])
                X_val[:,:d] = Scaler.transform(X_val[:,:d])
            ######## Feature Selection #############
            if feature_importance_list is not None:
                feature_importance = feature_importance_list[k]
                X_train = transform_data(X_train, feature_importance, num_features)
                X_val = transform_data(X_val, feature_importance, num_features)
            ####### Model training ##########
            y_pred = Regr.fit(X_train, y_train, y_train_log).predict(X_val, self.logtransform_Constant)

            mse_val_mean += mean_squared_error(y_val,y_pred)
        mse_val_mean = mse_val_mean/self.n_folds
        return np.sqrt(mse_val_mean)
        
    def TrainMyEnsembleOnAll(self, Regr, num_features, standardization=False):
        df_baseline = Generate_df_with_Dummies(self.df)
        len_of_dummies = df_baseline.shape[1] - (self.df.shape[1] - 2)
        y = df_baseline.pop('DepDelay').as_matrix()
        y_log = df_baseline.pop('LogDepDelay').as_matrix()
        X= df_baseline.as_matrix()
        d = X.shape[1]-len_of_dummies


        if standardization:
            Scaler = StandardScaler()
            Scaler.fit(X[:,:d])
            X[:,:d] = Scaler.transform(X[:,:d])
        
        from sklearn.ensemble import ExtraTreesRegressor
        
        ######## Feature Selection #############
        feature_extractor = ExtraTreesRegressor(n_estimators=100, n_jobs=4)
        feature_extractor.fit(X, y_log)
        feature_importance = feature_extractor.feature_importances_
        X = transform_data(X, feature_importance, num_features)
        ## Train
        y_fit = Regr.fit(X, y, y_log).predict(X, self.logtransform_Constant)
        return (Regr, feature_importance, y_fit)
        
    
    def PredictOnEvalData(self, Regr, standardization=False, feature_importance=None, 
                        num_features=None, in_original_order = False):
        
        df_with_dummies = self.df.copy()
        df_with_dummies.drop(['DepDelay','LogDepDelay'], axis=1, inplace=True)
        df_with_dummies = Generate_df_with_Dummies(df_with_dummies)
        df_eval_with_dummies = Generate_df_with_Dummies(self.df_eval, remove_infrequencies=False)
        Original_index = df_eval_with_dummies.pop('Original index')
        evalColumns = set(df_eval_with_dummies.columns.tolist())
        trainColumns = set(df_with_dummies.columns.tolist())
        to_exclude = evalColumns-trainColumns
        df_eval_with_dummies.drop(to_exclude, axis=1, inplace=True)
        evalColumns = set(df_eval_with_dummies.columns.tolist())        
        to_add = trainColumns - evalColumns
        for col in to_add:
            df_eval_with_dummies[col] = 0

        df_eval_with_dummies = df_eval_with_dummies[df_with_dummies.columns.tolist()]

        len_of_dummies = df_with_dummies.shape[1] - (self.df.shape[1] - 4)
        X_train = df_with_dummies.as_matrix()
        d = X_train.shape[1]-len_of_dummies
        if num_features is None:
            num_features = X_train.shape[1]
        X_eval = df_eval_with_dummies.as_matrix()
        

        if standardization:
            Scaler = StandardScaler()
            Scaler.fit(X_train[:,:d])
            X_eval[:,:d] = Scaler.transform(X_eval[:,:d])
            ######## Feature Selection #############
        if feature_importance is not None:
            X_eval = transform_data(X_eval, feature_importance, num_features)
            ####### Prediction ##########
        
        y_pred = Regr.predict(X_eval, self.logtransform_Constant)
        
        if in_original_order:
            temp = pd.DataFrame(Original_index)
            temp['y_pred'] = y_pred
            temp.sort('Original index', inplace=True)
            y_pred= temp['y_pred']
        return y_pred

class MyEnsemble(object):
    def __init__(self, RF, RF_RMSE, 
                       GB, GB_RMSE, 
                       Bagged_NN, Bagged_NN_RMSE):
        RMSEs = np.array([RF_RMSE,GB_RMSE,Bagged_NN_RMSE])
        c = RMSEs.min()-1
        
        
        self.RF = RF
        self.RF_weight = (1/(RF_RMSE-c)) / np.sum((1/(RMSEs-c)))
        
        self.GB = GB
        self.GB_weight = (1/(GB_RMSE-c)) / np.sum((1/(RMSEs-c)))
        
        self.Bagged_NN = Bagged_NN
        self.Bagged_NN_weight = (1/(Bagged_NN_RMSE-c)) / np.sum((1/(RMSEs-c)))
    
    def fit(self, X, y, y_log):
        self.RF.fit(X,y_log)
        self.GB.fit(X,y_log)
        self.Bagged_NN.fit(X,y)
        return self
    
    def predict(self,X,logtransform_Constant):
        
        y_pred_log_RF = self.RF.predict(X)
        y_pred_RF = np.exp(y_pred_log_RF) - logtransform_Constant
        
        y_pred_log_GB = self.GB.predict(X)
        y_pred_GB = np.exp(y_pred_log_GB) - logtransform_Constant
        
        y_pred_Bagged_NN = self.Bagged_NN.predict(X)
        
        y_pred = y_pred_RF*self.RF_weight + \
                 y_pred_GB*self.GB_weight + \
                 y_pred_Bagged_NN*self.Bagged_NN_weight
        
        return y_pred
                
        