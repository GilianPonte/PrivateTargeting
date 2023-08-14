def causal_neural_network(X, Y, T, scaling = False, simulations = 1, batch_size = 10, epochs = 50, folds = 10):
  from sklearn.linear_model import LogisticRegressionCV
  from keras.layers import Activation, LeakyReLU
  from keras import backend as K
  from keras.utils import get_custom_objects
  try:
    import keras_tuner
  except:
    print("installing keras tuner")
    !pip install keras_tuner -q
    import keras_tuner
  import random
  import tensorflow as tf
  from tensorflow import keras
  from keras import layers
  from sklearn.model_selection import KFold
  
  # reproducability
  random.seed(1)
  np.random.seed(1)
  tf.random.set_seed(1)
  
  # callback settings for early stopping and saving
  callback = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 25, mode = "min") # early stopping just like in rboost
  
  # define ate loss is equal to mean squared error between pseudo outcome and prediction of net.
  def ATE(y_true, y_pred):
    return tf.reduce_mean(y_pred, axis=-1)  # Note the `axis=-1`  

  # storage of cate estimates
  average_CATE_estimates_out_of_sample = []
  average_CATE_estimates_in_sample = []
  
  ## scale the data for well-behaved gradients
  if scaling == True:
    scaler0 = MinMaxScaler(feature_range = (-1, 1))
    scaler0 = scaler0.fit(X)
    X = scaler0.transform(X)

  ## Add leaky-relu so we can use it as a string
  get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})
  
  def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Choice(f"units_{i}", [2,4,8,16,32,64,128,256,512,1024]),
                activation=hp.Choice("activation", ["leaky-relu", "relu"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.8))
    model.add(layers.Dense(1, activation="linear"))
    #learning_rate = hp.Choice("lr", [1e-2,1e-3,1e-4]),

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["MSE"],
    )
    return model
  
  build_model(keras_tuner.HyperParameters())
  
  for i in range(0,simulations):
    print("iterations = " + str(i))
    random.seed(i)
    np.random.seed(i)
    tf.random.set_seed(i)

    # save models
    checkpoint_filepath_mx = 'm_x_'+ str(i) + '.hdf5'
    checkpoint_filepath_taux = 'tau_x' + str(i) + '.hdf5'
    mx_callbacks = [
      callback,
      tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath_mx,
      save_weights_only=False,
      monitor='val_loss',
      mode='min',
      save_freq="epoch",
      save_best_only=True),]
    tau_hat_callbacks = [
      callback,
      tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath_taux,
      save_weights_only=False,
      monitor='val_loss',
      mode='min',
      save_freq="epoch",
      save_best_only=True),]

    y_tilde_hat = [] # collect all the \tilde{Y}
    T_tilde_hat = [] # collect all the \tilde{T}
    callback = tf.keras.callbacks.EarlyStopping(monitor= "val_loss", patience = 20, mode = "min") # early stopping
    
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs=epochs,
        overwrite=True,
        directory="tuner",
        project_name="yhat",)
    
    if i == 0: # only cross-validate at first iteration, use same architecture subsequently.
      print("hyperparameter optimization for yhat")
      tuner.search(X, Y, epochs=epochs, validation_split=0.25, callbacks=[callback], verbose = 0)
      # Get the optimal hyperparameters
      best_hps=tuner.get_best_hyperparameters()[0]
      print(best_hps.values)
    
    cv = KFold(n_splits=folds) # K-fold validation
    
    for train_idx, test_idx in cv.split(X):
      print("training model for m(x)")
      model_m_x = tuner.hypermodel.build(best_hps)
      model_m_x.fit(
          X[train_idx],
          Y[train_idx],
          epochs = epochs,
          batch_size = batch_size,
          validation_data = (X[test_idx], Y[test_idx]),
          callbacks= mx_callbacks, # prevent overfitting with early stopping. If val_loss does not decrease after 10 epochs take that model.
          verbose = 0)
      model_m_x = tuner.hypermodel.build(best_hps)
      model_m_x.build(input_shape = (None,X.shape[1]))
      model_m_x.load_weights(checkpoint_filepath_mx)
      m_x = model_m_x.predict(x=X[test_idx]).reshape(len(Y[test_idx])) # obtain \hat{m}(x) from test set
      
      # obtain \tilde{Y} = Y_{i} - \hat{m}(x)
      print("obtaining Y_tilde")
      truth = Y[test_idx].T.reshape(len(Y[test_idx]))
      y_tilde = truth - m_x
      y_tilde_hat = np.concatenate((y_tilde_hat,y_tilde)) # cbind in r
      
      ## fit \hat{e}(x)
      print("training model for e(x)")
      clf = LogisticRegressionCV(cv=5, random_state=0).fit(X[train_idx], np.array(T[train_idx]).reshape(len(T[train_idx])))
      e_x = clf.predict_proba(X[test_idx]) # obtain \hat{e}(x)
      
      # obtain \tilde{T} = T_{i} - \hat{e}(x)
      print("obtaining T_tilde")
      truth = T[test_idx].T.reshape(len(T[test_idx]))
      T_tilde = truth - e_x[:,1]
      T_tilde_hat = np.concatenate((T_tilde_hat,T_tilde))

    CATE_estimates = []
    CATE = []

    epochs_train_loss_per_fold = []
    epochs_val_loss_per_fold = []

    pseudo_outcome = (y_tilde_hat/T_tilde_hat) # pseudo_outcome = \tilde{Y} / \tilde{T}

    ## weights
    w_weigths = np.square(T_tilde_hat) # \tilde{T}**2

    print("hyperparameter optimization for tau hat")
    tuner1 = keras_tuner.Hyperband(
      hypermodel=build_model,
      objective="val_loss",
      max_epochs=epochs,
      overwrite=True,
      directory="tuner",
      project_name="tau_hat",)
    
    if i == 0:
      tuner1.search(X, pseudo_outcome, epochs=epochs, validation_split=0.25, callbacks=[callback], verbose = 0)
      best_hps_tau =tuner1.get_best_hyperparameters()[0]
    
    cv = KFold(n_splits=folds)
    for train_idx, test_idx in cv.split(X):
      print("training for tau hat")
      tau_hat = tuner1.hypermodel.build(best_hps_tau)
      history_tau = tau_hat.fit(
          X[train_idx],
          pseudo_outcome[train_idx],
          sample_weight= w_weigths[train_idx],
          epochs = epochs,
          batch_size = batch_size,
          callbacks = tau_hat_callbacks,
          validation_data = (X[test_idx], pseudo_outcome[test_idx])
          )
      tau_hat = tuner1.hypermodel.build(best_hps_tau)
      tau_hat.build(input_shape = (None,X.shape[1]))
      tau_hat.load_weights(checkpoint_filepath_taux)
      CATE = tau_hat.predict(x=X[test_idx]).reshape(len(X[test_idx]))

      #print("average treatment effect of = " + str(np.mean(CATE)))
      CATE_estimates = np.concatenate((CATE_estimates,CATE)) # store CATE's
    #print(np.mean(CATE_estimates))
    average_CATE_estimates_out_of_sample = np.append(average_CATE_estimates_out_of_sample,np.mean(CATE_estimates))

    tau_hat_final = tuner1.hypermodel.build(best_hps_tau)
    tau_hat.build(input_shape = (None,X.shape[1]))

    print("training for tau hat")
    tau_hat_final.fit(
          X,
          pseudo_outcome,
          sample_weight= w_weigths,
          epochs = best_hps_tau.values['tuner/epochs'],
          batch_size = batch_size
          #callbacks = [callback]
          )
    CATE = tau_hat_final.predict(x=X).reshape(len(X))
    average_CATE_estimates_in_sample = np.append(average_CATE_estimates_in_sample,np.mean(CATE))
  return average_CATE_estimates_in_sample, average_CATE_estimates_out_of_sample
