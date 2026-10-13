from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
import seaborn as sns
import tensorflow_docs as tfdocs

#this file has the functions of when I was working with the uber movement dataset

#function for returning the neural network model
def create_nn_model(init_mode='normal', activation='relu', dropout_rate=0.0, weight_constraint=0):

    nn_model = Sequential()
    nn_model.add(Dense(64, input_dim=10, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    nn_model.add(Dropout(dropout_rate))
    nn_model.add(Dense(1, kernel_initializer=init_mode))
    #nn_model.add(Dense(1, activation="linear")) #regressor neuron?
    nn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    
    return nn_model

def get_num_lanes(dict_edge):

    num_lanes = np.nan
    try:

        if type(dict_edge['lanes']) is not list:
            num_lanes = dict_edge['lanes']
            if num_lanes.isdigit():
                num_lanes = int(num_lanes)
                return num_lanes
        else:
            avg_num_lanes = 0
            for lanes in dict_edge['lanes']:
                if lanes.isdigit():
                    avg_num_lanes = avg_num_lanes + int(lanes)
            avg_num_lanes = int(avg_num_lanes/len(dict_edge['lanes']))
            if avg_num_lanes > 0:
                num_lanes = avg_num_lanes
                return num_lanes

    except KeyError:
        pass

    return num_lanes

def get_highway_info(dict_edge):

    try:

        if type(dict_edge['highway']) is not list:
            if dict_edge['highway'] == 'motorway' or dict_edge['highway'] == 'motorway_link':
                return 1
            if dict_edge['highway'] == 'trunk' or dict_edge['highway'] == 'trunk_link':
                return 2
            if dict_edge['highway'] == 'primary' or dict_edge['highway'] == 'primary_link':
                return 3
            if dict_edge['highway'] == 'secondary' or dict_edge['highway'] == 'secondary_link': 
                return 4
            if dict_edge['highway'] == 'tertiary' or dict_edge['highway'] == 'tertiary_link':
                return 5
            if dict_edge['highway'] == 'unclassified':
                return 6
            if dict_edge['highway'] == 'residential':
                return 7
            if dict_edge['highway'] == 'living_street':
                return 8
            if dict_edge['highway'] == 'service':
                return 9
            if dict_edge['highway'] == 'pedestrian':
                return 10
            if dict_edge['highway'] == 'track':
                return 11
            if dict_edge['highway'] == 'road':
                return 12

        else:
            #print(dict_edge['highway'])
            pass

    except KeyError:
        pass

    return np.nan

def get_uber_speed_data_prediction_groupby(G_drive, speed_data):

    api = osm.OsmApi()
    #load speed data from csv files
    path = speed_data
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    uber_data = pd.concat(df_from_each_file, ignore_index=True)

    #uber_data = pd.read_csv("../uber_movement/movement-speeds-hourly-cincinnati-2019-10.csv")
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    #print(concatenated_df.head())
    #print(uber_data.head())
    
    print("start number of rows", len(uber_data))

    nodes_in_graph = [] #nodes that are mapped by the uber movement speed database
    for node in unique_nodes:
        if node in G_drive.nodes():
            if node not in nodes_in_graph:
                nodes_in_graph.append(node)

    uber_data = uber_data[uber_data['osm_start_node_id'].isin(nodes_in_graph) & uber_data['osm_end_node_id'].isin(nodes_in_graph)]
    
    print("mid number of rows", len(uber_data))

    #talvez pegar o dia e fazer aqui seg, terça etc e dar groupby com isso tb?
    
    #add day of the week (monday, tuesday, wednesday, thursday, friday, saturday, sunday etc) info
    #add new column? job day? weekend? holiday?
    #these columns are created based on info that might help, such as peak hours etc
    unique_days = pd.unique(uber_data[['day']].values.ravel('K'))
    unique_months = pd.unique(uber_data[['month']].values.ravel('K'))
    unique_years = pd.unique(uber_data[['year']].values.ravel('K'))

    #add day info
    uber_data["week_day"] = np.nan
    for year in unique_years:
        for month in unique_months:
            for day in unique_days:
                try:
                    ans = datetime.date(year, month, day).weekday()
                    uber_data.loc[(uber_data['day'] == day) & (uber_data['month'] == month) & (uber_data['year'] == year), 'week_day'] = ans
                except ValueError:
                    pass
    

    uber_data = uber_data.groupby(['osm_start_node_id','osm_end_node_id', 'hour', 'week_day'], as_index=False)['speed_mph_mean'].mean()
    uber_data = pd.DataFrame(uber_data)
    
    print("mid number of rows (after grouby)", len(uber_data))

    print(uber_data.head())
    print(list(uber_data.columns))

    #add lat/lon info and max speed
    #in this part info from openstreetmaps is added
    uber_data["start_node_y"] = np.nan
    uber_data["start_node_x"] = np.nan
    uber_data["end_node_y"] = np.nan
    uber_data["end_node_x"] = np.nan
    uber_data["max_speed_mph"] = np.nan
    uber_data["num_lanes"] = np.nan
    uber_data["highway"] = np.nan

    #unique_highway_str = pd.unique(uber_data[['highway']].values.ravel('K'))
    #print(unique_highway_str)

    for (u,v,k) in G_drive.edges(data=True):
        try:
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = G_drive.nodes[u]['y']
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = G_drive.nodes[u]['x']

            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = G_drive.nodes[v]['y']
            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = G_drive.nodes[v]['x']

            if (uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] == 0.0).all():
                nodeu = api.NodeGet(u)
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = nodeu['lat']
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = nodeu['lon']

            if (uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] == 0.0).all():
                nodev = api.NodeGet(v)
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = nodev['lat']
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = nodev['lon']

        except KeyError:
            pass

        #add atribute max speed 
        dict_edge = {}
        dict_edge = G_drive.get_edge_data(u, v)
        dict_edge = dict_edge[0]
        
        max_speed = get_max_speed_road(dict_edge)
        num_lanes = get_num_lanes(dict_edge)
        #highway
        highway_type = get_highway_info(dict_edge)
        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'max_speed_mph'] = max_speed

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'num_lanes'] = num_lanes

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'highway'] = highway_type

    #print("min hour", uber_data["hour"].min())
    #print("max hour", uber_data["hour"].max())
    
    uber_data["period_day"] = np.nan
    uber_data.loc[(uber_data['hour'] >= 0) & (uber_data['hour'] <= 6), 'period_day'] = 1 #before peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 7) & (uber_data['hour'] <= 9), 'period_day'] = 2 #peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 10) & (uber_data['hour'] <= 16), 'period_day'] = 3 #before peak hours - afternoon
    uber_data.loc[(uber_data['hour'] >= 17) & (uber_data['hour'] <= 20), 'period_day'] = 4 #peak hours - afternoon 
    uber_data.loc[(uber_data['hour'] >= 21) & (uber_data['hour'] <= 23), 'period_day'] = 5 # night period
    
    #upstream and downstream roads??
    #uber_data["adjacent_roads_speed"] = np.nan

    print(uber_data.isna().sum())
    #clean NA values
    print("clean NA values")
    uber_data = uber_data.dropna()

    uber_data["num_lanes"] = pd.to_numeric(uber_data["num_lanes"])

    #print("min lanes", uber_data["num_lanes"].min())
    #print("max lanes", uber_data["num_lanes"].max())

    #print("min highway", uber_data["highway"].min())
    #print("max highway", uber_data["highway"].max())

    print("end number of rows", len(uber_data))

    print(list(uber_data.columns))
    #sns.pairplot(uber_data[["max_speed_mph", "num_lanes", "highway"]], diag_kind="kde")
    
    #print(uber_data.head())
    print(uber_data.dtypes)        
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    #scaler = MinMaxScaler(feature_range=(0, 1))

    #columns with groupby
    #0 ID
    #1 ID
    #2 'hour'
    #3 day of the week (monday...)
    #4 'speed' (TARGET)
    #5,6,7,8 lat/lon of nodes representing the road
    #9 - max_speed_mph
    #10 - number of lanes
    #11 - highway
    #12 -  period of the day
    
    #divide in attributes and labels
    X = uber_data.iloc[:, [2,3,5,6,7,8,9,10,11,12]].values
    y = uber_data.iloc[:, 4].values 
    scaler = StandardScaler()

    #columns without groupby
    #0 - year
    #1 - month
    #2 - day
    #3 - hour
    #4 - utc_timestamp
    #5,6,7,8,9,10 - IDs
    #11 - speed (TARGET)
    #12 - speed std deviation
    #13, 14, 15, 16 - lat/lon of nodes representing the road
    #17 - max_speed_mph
    #18 - number of lanes
    #19 - highway
    #20 - day of the week (monday...)
    #21 - period of the day
    
    #divide in attributes and labels
    #X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    #y = uber_data.iloc[:, 11].values 
    #scaler = StandardScaler()
    
    '''
    #knn kfold
    print("knn start")
    k_scores = []
    best_score = 999999
    best_k = -1
    for k in range(40):
        k = k+1
        knn_reg = KNeighborsRegressor(n_neighbors=k)
        regressor = make_pipeline(scaler, knn_reg)
        scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
        k_scores.append(scores_mean)
        if scores_mean < best_score:
            best_k = k
            best_score = scores_mean
    print(k_scores)
    print("best k, best msqerror:", best_k, best_score)
    ''' 

    '''
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    #linear regression
    lin_reg = LinearRegression()
    regressor = make_pipeline(scaler, lin_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('linear regression msqerror:', scores_mean)
    '''

    '''   
    #SVR
    #gamma - scale/auto/0.1
    print('SVR start')
    #srv_rbf = SVR(kernel='rbf', gamma='scale', C=1.57, epsilon=0.03)
    srv_rbf = SVR(kernel='rbf', gamma='auto')
    #srv_linear = SVR(kernel='linear')
    regressor = make_pipeline(scaler, srv_rbf)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('SVR msqerror:', scores_mean)
    

    
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    '''

    #neural network
    print('nn start')
    estimators = []
    estimators.append(('standardize', scaler))
    #validation_split=0.2 -> testar com validation split?
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    #estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=5, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])))
    estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=5, verbose=0)))
    regressor = Pipeline(estimators)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('neural network msqerror:', scores_mean)
    

    '''
    model = create_nn_model()
    print(model.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=5, epochs=100)
    #validation_split=0.2
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=5, epochs=100)
    y_pred = model.predict(X_test)
    #rmserror = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    msqerror = mean_squared_error(y_test,y_pred) #calculate msqerror
    print('neural network msqerror:', msqerror)
    '''

    '''
    #hyperparameter optimization technique usind Grid Search
    #The best_score_ member provides access to the best score observed during the optimization procedure 
    #the best_params_ describes the combination of parameters that achieved the best results
    print('grid search SVM')
    svmr = SVR()
    pipe = Pipeline([('scale', scaler),('svm', svmr)])
    #define the grid search parameters
    param_grid = [{'svm__kernel': ['rbf', 'poly', 'sigmoid'],'svm__C': [0.1, 1, 10, 100],'svm__gamma': [1,0.1,0.01,0.001],},]
    #param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #param_grid = {'C': [1], 'gamma': [0.1],'kernel': ['rbf']}
    gd_svr = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring="neg_mean_squared_error",cv=10,n_jobs=-1,return_train_score=False,refit=True)
    #pipe_svm = make_pipeline(scaler, gd_sr)
    grid_svr_result = gd_svr.fit(X,y)
    print(grid_svr_result.cv_results_)
    print(grid_svr_result.best_estimator_)
    '''

    '''
    #define the grid search parameters
    #Tune Batch Size and Number of Epochs
    batch_size = [5, 10, 20, 40]
    epochs = [10, 50, 100, 200, 400]
    #Tune the Training Optimization Algorithm => optimization algorithm used to train the network, each with default parameters.
    #often you will choose one approach a priori and instead focus on tuning its parameters on your problem
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #Tune Learning Rate and Momentum <- relacionado ao algoritmo selecionado anteriormente
    #Tune Network Weight Initialization
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #Tune the Neuron Activation Function
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    #Tune Dropout Regularization
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #Tune the Number of Neurons in the Hidden Layer
    neurons = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid_nn = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_nn_result = grid_nn.fit(X, Y)
    print(grid_svr_result.cv_results_)
    print(grid_svr_result.best_estimator_)
    '''
    

    '''
    plt.plot(y_test, color = 'red', label = 'Real data')
    plt.plot(y_pred, color = 'blue', label = 'Predicted data')
    plt.title('Prediction')
    plt.legend()
    plt.show()
    '''

    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    ''' 

    '''
    plt.scatter(y_test, y_pred)

    plt.xlabel('True Values')

    plt.ylabel('Predictions')
    '''

    '''
    #logistic regression
    log_reg = LogisticRegression()
    regressor = make_pipeline(scaler, log_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('logistic regression msqerror:', scores_mean)
    '''

    #other error calculating. but i think those are not good for knn
    #print(np.mean(y_pred != y_test))
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    
    #add new columns ->doutros atributos da aresta q tenha do osmnx => for this we need to deal with some roads that don't have the info on max speed, etc
    #do the prediction on the missing roads

def get_uber_speed_data_prediction(G_drive, speed_data):

    api = osm.OsmApi()
    #load speed data from csv files
    path = speed_data
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    uber_data = pd.concat(df_from_each_file, ignore_index=True)


    #uber_data = pd.read_csv("../uber_movement/movement-speeds-hourly-cincinnati-2019-10.csv")
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    #print(concatenated_df.head())
    #print(uber_data.head())
    
    print("start number of rows", len(uber_data))

    nodes_in_graph = [] #nodes that are mapped by the uber movement speed database
    for node in unique_nodes:
        if node in G_drive.nodes():
            if node not in nodes_in_graph:
                nodes_in_graph.append(node)

    uber_data = uber_data[uber_data['osm_start_node_id'].isin(nodes_in_graph) & uber_data['osm_end_node_id'].isin(nodes_in_graph)]
    
    print("mid number of rows", len(uber_data))

    #add lat/lon info and max speed
    #in this part info from openstreetmaps is added
    uber_data["start_node_y"] = np.nan
    uber_data["start_node_x"] = np.nan
    uber_data["end_node_y"] = np.nan
    uber_data["end_node_x"] = np.nan
    uber_data["max_speed_mph"] = np.nan
    uber_data["num_lanes"] = np.nan
    uber_data["highway"] = np.nan

    #unique_highway_str = pd.unique(uber_data[['highway']].values.ravel('K'))
    #print(unique_highway_str)

    for (u,v,k) in G_drive.edges(data=True):
        try:
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = G_drive.nodes[u]['y']
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = G_drive.nodes[u]['x']

            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = G_drive.nodes[v]['y']
            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = G_drive.nodes[v]['x']

            if (uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] == 0.0).all():
                nodeu = api.NodeGet(u)
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = nodeu['lat']
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = nodeu['lon']

            if (uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] == 0.0).all():
                nodev = api.NodeGet(v)
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = nodev['lat']
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = nodev['lon']

        except KeyError:
            pass

        #add atribute max speed 
        dict_edge = {}
        dict_edge = G_drive.get_edge_data(u, v)
        dict_edge = dict_edge[0]
        
        max_speed = get_max_speed_road(dict_edge)
        num_lanes = get_num_lanes(dict_edge)
        #highway
        highway_type = get_highway_info(dict_edge)
        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'max_speed_mph'] = max_speed

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'num_lanes'] = num_lanes

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'highway'] = highway_type

    
    
    #add day of the week (monday, tuesday, wednesday, thursday, friday, saturday, sunday etc) info
    #add new column? job day? weekend? holiday?
    #these columns are created based on info that might help, such as peak hours etc
    unique_days = pd.unique(uber_data[['day']].values.ravel('K'))
    unique_months = pd.unique(uber_data[['month']].values.ravel('K'))
    unique_years = pd.unique(uber_data[['year']].values.ravel('K'))

    #add day info
    uber_data["week_day"] = np.nan
    for year in unique_years:
        for month in unique_months:
            for day in unique_days:
                try:
                    ans = datetime.date(year, month, day).weekday()
                    uber_data.loc[(uber_data['day'] == day) & (uber_data['month'] == month) & (uber_data['year'] == year), 'week_day'] = ans
                except ValueError:
                    pass

    #print("min hour", uber_data["hour"].min())
    #print("max hour", uber_data["hour"].max())
    
    uber_data["period_day"] = np.nan
    uber_data.loc[(uber_data['hour'] >= 0) & (uber_data['hour'] <= 6), 'period_day'] = 1 #before peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 7) & (uber_data['hour'] <= 9), 'period_day'] = 2 #peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 10) & (uber_data['hour'] <= 16), 'period_day'] = 3 #before peak hours - afternoon
    uber_data.loc[(uber_data['hour'] >= 17) & (uber_data['hour'] <= 20), 'period_day'] = 4 #peak hours - afternoon 
    uber_data.loc[(uber_data['hour'] >= 21) & (uber_data['hour'] <= 23), 'period_day'] = 5 # night period
    
    #upstream and downstream roads??
    #uber_data["adjacent_roads_speed"] = np.nan


    print(uber_data.isna().sum())
    #clean NA values
    print("clean NA values")
    uber_data = uber_data.dropna()

    uber_data["num_lanes"] = pd.to_numeric(uber_data["num_lanes"])

    #print("min lanes", uber_data["num_lanes"].min())
    #print("max lanes", uber_data["num_lanes"].max())

    #print("min highway", uber_data["highway"].min())
    #print("max highway", uber_data["highway"].max())

    print("end number of rows", len(uber_data))

    

    print(list(uber_data.columns))
    #sns.pairplot(uber_data[["max_speed_mph", "num_lanes", "highway"]], diag_kind="kde")
    
    print(uber_data.head())
    print(uber_data.dtypes)        
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    #scaler = MinMaxScaler(feature_range=(0, 1))

    #columns without groupby
    #0 - year
    #1 - month
    #2 - day
    #3 - hour
    #4 - utc_timestamp
    #5,6,7,8,9,10 - IDs
    #11 - speed (TARGET)
    #12 - speed std deviation
    #13, 14, 15, 16 - lat/lon of nodes representing the road
    #17 - max_speed_mph
    #18 - number of lanes
    #19 - highway
    #20 - day of the week (monday...)
    #21 - period of the day
    
    #divide in attributes and labels
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    
    '''
    #knn kfold
    k_scores = []
    best_score = 999999
    best_k = -1
    for k in range(40):
        k = k+1
        knn_reg = KNeighborsRegressor(n_neighbors=k)
        regressor = make_pipeline(scaler, knn_reg)
        scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
        k_scores.append(scores_mean)
        if scores_mean < best_score:
            best_k = k
            best_score = scores_mean
    print(k_scores)
    print("best k, best msqerror:", best_k, best_score)
    '''

    '''
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    #linear regression
    lin_reg = LinearRegression()
    regressor = make_pipeline(scaler, lin_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('linear regression msqerror:', scores_mean)
    '''

    '''
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    #SVR
    #gamma - scale/auto/0.1
    #srv_rbf = SVR(kernel='rbf', gamma='scale', C=1.57, epsilon=0.03)
    srv_rbf = SVR(kernel='rbf', gamma='auto')
    #srv_linear = SVR(kernel='linear')
    regressor = make_pipeline(scaler, srv_rbf)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('SVR msqerror:', scores_mean)
    '''
    
    '''
    #X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    #y = uber_data.iloc[:, 11].values 
    #scaler = StandardScaler()
    #neural network
    estimators = []
    estimators.append(('standardize', scaler))
    #validation_split=0.2 -> testar com validation split?
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    #estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=5, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])))
    estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=10, verbose=0)))
    regressor = Pipeline(estimators)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('neural network msqerror:', scores_mean)
    '''

    '''
    model = create_nn_model()
    print(model.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=5, epochs=100)
    #validation_split=0.2
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=5, epochs=100)
    y_pred = model.predict(X_test)
    #rmserror = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    msqerror = mean_squared_error(y_test,y_pred) #calculate msqerror
    print('neural network msqerror:', msqerror)
    '''

    '''
    #hyperparameter optimization technique usind Grid Search
    #The best_score_ member provides access to the best score observed during the optimization procedure 
    #the best_params_ describes the combination of parameters that achieved the best results
    print('grid search SVM')
    svmr = SVR()
    pipe = Pipeline([('scale', scaler),('svm', svmr)])
    #define the grid search parameters
    param_grid = [{'svm__kernel': ['rbf', 'poly', 'sigmoid'],'svm__C': [0.1, 1, 10, 100],'svm__gamma': [1,0.1,0.01,0.001],},]
    #param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #param_grid = {'C': [1], 'gamma': [0.1],'kernel': ['rbf']}
    gd_svr = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring="neg_mean_squared_error",cv=3,n_jobs=-1,return_train_score=False,refit=True)
    #pipe_svm = make_pipeline(scaler, gd_sr)
    grid_svr_result = gd_svr.fit(X,y)
    print(grid_svr_result.cv_results_)
    print(grid_svr_result.best_estimator_)
    '''

    
    print('NEURAL NETWORK GRID SEARCH - BATCH SIZE AND EPOCHS')
    #define the grid search parameters
    #Tune Batch Size and Number of Epochs
    
    #batch_size = [5, 8, 10, 16, 20]
    #epochs = [100, 200, 400, 800, 1600]
    batch_size = [16]
    epochs = [800]

    #Tune the Training Optimization Algorithm => optimization algorithm used to train the network, each with default parameters.
    #often you will choose one approach a priori and instead focus on tuning its parameters on your problem
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #Tune Learning Rate and Momentum <- relacionado ao algoritmo selecionado anteriormente
    #Tune Network Weight Initialization
    
    #init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    init_mode = ['he_uniform']

    #Tune the Neuron Activation Function
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    #Tune Dropout Regularization
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #Tune the Number of Neurons in the Hidden Layer
    neurons = [1, 5, 10, 15, 20, 25, 30]
    
    param_grid = dict(batch_size=batch_size, epochs=epochs, init_mode=init_mode, activation=activation, weight_constraint=weight_constraint, dropout_rate=dropout_rate)
    nn_model = KerasRegressor(build_fn=create_nn_model, verbose=0)
    grid_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid, n_jobs=-1, cv=3)
    X = scaler.fit_transform(X)
    grid_nn_result = grid_nn.fit(X, y)
    print(grid_nn_result.cv_results_)
    #print(grid_nn_result.best_estimator_)
    print("Best: %f using %s" % (grid_nn_result.best_score_, grid_nn_result.best_params_))
    
    '''
    plt.plot(y_test, color = 'red', label = 'Real data')
    plt.plot(y_pred, color = 'blue', label = 'Predicted data')
    plt.title('Prediction')
    plt.legend()
    plt.show()
    '''

    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    ''' 

    '''
    plt.scatter(y_test, y_pred)

    plt.xlabel('True Values')

    plt.ylabel('Predictions')
    '''

    '''
    #logistic regression
    log_reg = LogisticRegression()
    regressor = make_pipeline(scaler, log_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('logistic regression msqerror:', scores_mean)
    '''

    #other error calculating. but i think those are not good for knn
    #print(np.mean(y_pred != y_test))
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    
    #add new columns ->doutros atributos da aresta q tenha do osmnx => for this we need to deal with some roads that don't have the info on max speed, etc
    #do the prediction on the missing roads

def get_uber_speed_data_mean(G_drive, speed_data, day_of_the_week):
    
    #function returns the avg speed for each road

    #load speed data from csv files
    path = speed_data
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    uber_data = pd.concat(df_from_each_file, ignore_index=True)
    
    count_nodes = 0
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    nodes_in_graph = []
    for node in unique_nodes:
        if node in G_drive.nodes():
            if node not in nodes_in_graph:
                nodes_in_graph.append(node)

    uber_data = uber_data[uber_data['osm_start_node_id'].isin(nodes_in_graph) & uber_data['osm_end_node_id'].isin(nodes_in_graph)]
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    #uber_data_nd = uber_data[['osm_start_node_id','osm_end_node_id']].drop_duplicates()
    #unique_nodes_nd = pd.unique(uber_data_nd[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))

    unique_days = pd.unique(uber_data[['day']].values.ravel('K'))
    unique_months = pd.unique(uber_data[['month']].values.ravel('K'))
    unique_years = pd.unique(uber_data[['year']].values.ravel('K'))

    #add day info
    uber_data["week_day"] = np.nan
    for year in unique_years:
        for month in unique_months:
            for day in unique_days:
                try:
                    ans = datetime.date(year, month, day).weekday()
                    uber_data.loc[(uber_data['day'] == day) & (uber_data['month'] == month) & (uber_data['year'] == year), 'week_day'] = ans
                except ValueError:
                    pass

    uber_data = uber_data.loc[uber_data["week_day"] == day_of_the_week]
    #this value is used to add to roads in which speed information is unkown
    speed_mean_overall = uber_data['speed_mph_mean'].mean()

    speed_avg_data = uber_data.groupby(['osm_start_node_id','osm_end_node_id', 'hour'], as_index=False)['speed_mph_mean'].mean()

    

    #speed_mean_overall = speed_avg_data['speed_mph_mean'].mean()

    return speed_avg_data, speed_mean_overall

    
    #speed_avg_data.columns = ['osm_start_node_id','osm_end_node_id', 'hour', 'avg_speed']
    #print(speed_avg_data.head())

    #plot network to show nodes that are in the uber speed data
    #nc = ['r' if (node in unique_nodes) else '#336699' for node in G_drive.nodes()]
    #ns = [12 if (node in unique_nodes) else 6 for node in G_drive.nodes()]
    #fig, ax = ox.plot_graph(G_drive, node_size=ns, node_color=nc, node_zorder=2, save=True, filename='cincinnati_with_nodes_speed')
    
    '''
    for (u,v,k) in G_drive.edges(data=True):
        #print (u,v,k)
        try:
            G_drive[u][v][0]['uberspeed'] = 0
            G_drive[u][v][0]['num_occur'] = 0
        except KeyError:
            pass

    
    for index, row in uber_data.iterrows():
        try:
            u = row['osm_start_node_id']
            v = row['osm_end_node_id']
            G_drive[u][v][0]['uberspeed'] = G_drive[u][v][0]['uberspeed'] + row['speed_mph_mean']
            G_drive[u][v][0]['num_occur'] = G_drive[u][v][0]['num_occur'] + 1
        except KeyError:
            pass
    
    for (u,v,k) in G_drive.edges(data=True):
        if G_drive[u][v][0]['num_occur'] > 0:
            G_drive[u][v][0]['uberspeed'] = (G_drive[u][v][0]['uberspeed']/G_drive[u][v][0]['num_occur'])
            #G_drive[u][v][0]['num_occur'] = 0
    '''