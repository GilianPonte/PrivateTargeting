def causal_neural_network(X, Y, T, scaling = True, simulations = 1, batch_size = 100, epochs = 100, max_epochs = 10, folds = 5, directory = "tuner"):
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

  # callback settings for early stopping and saving
  callback = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 5, mode = "min") # early stopping just like in rboost

  # define ate loss is equal to mean squared error between pseudo outcome and prediction of net.
  def ATE(y_true, y_pred):
    return tf.reduce_mean(y_pred, axis=-1)  # Note the `axis=-1`

  # storage of cate estimates
  average_treatment_effect = []

  ## scale the data for well-behaved gradients
  if scaling == True:
    scaler0 = MinMaxScaler(feature_range = (-1, 1))
    scaler0 = scaler0.fit(X)
    X = scaler0.transform(X)
    X = pd.DataFrame(X)

  # get index to speed up tuning
  idx = pd.DataFrame(pd.DataFrame(X).index).sample(1000).index
  Y_tuning = pd.DataFrame(Y).iloc[idx]
  X_tuning = pd.DataFrame(X).iloc[idx,:]

  ## Add leaky-relu so we can use it as a string
  get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})

  def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X.shape[1],)))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Choice(f"units_{i}", [8, 16, 32, 64,256,512,1024, 2048, 4096]),
                activation=hp.Choice("activation", ["leaky-relu", "relu"]),
            )
        )
    model.add(layers.Dense(1, activation="linear"))
    #learning_rate = hp.Choice("lr", [1e-2,1e-3,1e-4]),

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["MSE"],
    )
    return model

  for i in range(0,simulations):
    print("iteration = " + str(i+1))
    random.seed(i)
    np.random.seed(i)
    tf.random.set_seed(i)

    # save models
    checkpoint_filepath_mx = 'm_x_'+ str(i+1) + '.hdf5'
    checkpoint_filepath_taux = 'tau_x' + str(i+1) + '.hdf5'
    mx_callbacks = [callback, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_mx, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True),]
    tau_hat_callbacks = [callback, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_taux, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True),]

    y_tilde_hat = [] # collect all the \tilde{Y}
    T_tilde_hat = [] # collect all the \tilde{T}
    m_x_hat = [] # collect all m_x_hat for print
    e_x_hat = [] # collect all e_x_hat for print

    if i == 0: # only cross-validate at first iteration, use same architecture subsequently.
      print("hyperparameter optimization for yhat")
      tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs= max_epochs,
        overwrite=True,
        directory=directory,
        project_name="yhat",)
      tuner.search(X_tuning, Y_tuning, epochs = epochs, validation_split=0.25, verbose = 1)
      # Get the optimal hyperparameters
      best_hps=tuner.get_best_hyperparameters()[0]
      print("the optimal architecture is: " + str(best_hps.values))

    cv = KFold(n_splits=folds, shuffle = False) # K-fold validation shuffle is off to prevent additional noise?

    for k, (train_idx, test_idx) in enumerate(cv.split(X)):
      #print("training model for m(x)")
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
      m_x = model_m_x.predict(x=X[test_idx], verbose = 0).reshape(len(Y[test_idx])) # obtain \hat{m}(x) from test set

      # obtain \tilde{Y} = Y_{i} - \hat{m}(x)
      #print("obtaining Y_tilde")
      truth = Y[test_idx].T.reshape(len(Y[test_idx]))
      y_tilde = truth - m_x
      y_tilde_hat = np.concatenate((y_tilde_hat,y_tilde)) # cbind in r
      m_x_hat = np.concatenate((m_x_hat,m_x)) # cbind in r

      ## fit \hat{e}(x)
      #print("training model for e(x)")
      clf = LogisticRegression(verbose = 0).fit(X[train_idx], np.array(T[train_idx]).reshape(len(T[train_idx])))
      e_x = clf.predict_proba(X[test_idx]) # obtain \hat{e}(x)
      print(f"Fold {k}: mean(m_x) = " + str(np.round(np.mean(m_x),2)) + ", sd(m_x) = " + str(np.round(np.std(m_x),3)) + " and mean(e_x) = " + str(np.round(np.mean(e_x[:,1]),2)) + ", sd(e_x) = " + str(np.round(np.std(e_x[:,1]),3)))

      # obtain \tilde{T} = T_{i} - \hat{e}(x)
      #print("obtaining T_tilde")
      truth = T[test_idx].T.reshape(len(T[test_idx]))
      T_tilde = truth - e_x[:,1]
      T_tilde_hat = np.concatenate((T_tilde_hat,T_tilde))
      e_x_hat = np.concatenate((e_x_hat,e_x[:,1]))

    print("mean(m_x) = " + str(np.round(np.mean(m_x_hat),2)) + ", sd(m_x) = " + str(np.round(np.std(m_x_hat),3)) + " and mean(e_x) = " + str(np.round(np.mean(e_x_hat),2)) + ", sd(e_x) = " + str(np.round(np.std(e_x_hat),3)))
    # storage
    CATE_estimates = []
    CATE = []

    ## pseudo_outcome and weights
    pseudo_outcome = (y_tilde_hat/T_tilde_hat) # pseudo_outcome = \tilde{Y} / \tilde{T}
    w_weigths = np.square(T_tilde_hat) # \tilde{T}**2

    if i == 0:
      print("hyperparameter optimization for tau hat")
      tuner1 = keras_tuner.Hyperband(
          hypermodel=build_model,
          objective="val_loss",
          max_epochs=max_epochs,
          overwrite=True,
          directory=directory,
          project_name="tau_hat",)
      tuner1.search(X[idx,:], pseudo_outcome[idx], epochs=epochs, validation_split=0.25, verbose = 1)
      best_hps_tau =tuner1.get_best_hyperparameters()[0]
      print("the optimal architecture is: " + str(best_hps_tau.values))

    cv = KFold(n_splits=folds, shuffle = False)
    print("training for tau hat")
    for  k, (train_idx, test_idx) in enumerate(cv.split(X)):

      tau_hat = tuner1.hypermodel.build(best_hps_tau)
      history_tau = tau_hat.fit(
          X[train_idx],
          pseudo_outcome[train_idx],
          sample_weight= w_weigths[train_idx],
          epochs = epochs,
          batch_size = batch_size,
          callbacks = tau_hat_callbacks,
          validation_data = (X[test_idx], pseudo_outcome[test_idx]),
          verbose = 0)
      tau_hat = tuner1.hypermodel.build(best_hps_tau)
      tau_hat.build(input_shape = (None,X.shape[1]))
      tau_hat.load_weights(checkpoint_filepath_taux)
      CATE = tau_hat.predict(x=X[test_idx], verbose = 0).reshape(len(X[test_idx]))
      print(f"Fold {k}: mean(tau_hat) = " + str(np.round(np.mean(CATE),2)) + ", sd(m_x) = " + str(np.round(np.std(CATE),3)))

      CATE_estimates = np.concatenate((CATE_estimates,CATE)) # store CATE's
    average_treatment_effect = np.append(average_treatment_effect, np.mean(CATE_estimates))
    print("ATE = " + str(np.round(np.mean(average_treatment_effect), 4)) + ", std(ATE) = " + str(np.round(np.std(average_treatment_effect), 3)))

  return average_treatment_effect, CATE_estimates, tau_hat
