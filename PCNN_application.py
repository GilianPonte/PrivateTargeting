def private_causal_neural_network(X, Y, T, scaling = True, simulations = 1, batch_size = 100, epochs = 100, max_epochs = 10, folds = 2, directory = "tuner", noise_multiplier = 1, propensity = 0):
  from sklearn.linear_model import LogisticRegressionCV
  from keras.layers import Activation, LeakyReLU
  from keras import backend as K
  import tensorflow
  from tensorflow.keras.utils import get_custom_objects
  import math
  try:
    import tensorflow_privacy
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
  except:
    print("installing tensorflow privacy")
    !pip install tensorflow-probability -q
    !pip install tensorflow_privacy -q
    import tensorflow_privacy
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
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
  from tensorflow.keras import regularizers

  # calculate epsilon
  print(tensorflow_privacy.compute_dp_sgd_privacy_statement(number_of_examples = len(X), batch_size = batch_size, noise_multiplier = noise_multiplier, num_epochs = epochs, delta = 1/len(X),
                                                            used_microbatching = True, max_examples_per_user = 1))

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
        optimizer= tensorflow.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["MSE"],
    )
    return model

  for i in range(0,simulations):
    print("iteration = " + str(i+1))
    random.seed(i)
    np.random.seed(i)
    tf.random.set_seed(i)

    # for epsilon calculation
    idx = np.random.permutation(pd.DataFrame(X).index)
    X = np.array(pd.DataFrame(X).reindex(idx))
    Y = np.array(pd.DataFrame(Y).reindex(idx))
    T = np.array(pd.DataFrame(T).reindex(idx))

    # save models
    checkpoint_filepath_mx = 'm_x_'+ str(i+1) + ".hdf5"
    checkpoint_filepath_taux = 'tau_x' + str(i+1) + ".hdf5"
    mx_callbacks = [callback, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_mx, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True),]
    tau_hat_callbacks = [callback, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_taux, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True),]
    tau_hat_callbacks_insample = [callback, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_taux, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True),]

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
      tuner.search(X, Y, epochs = epochs, validation_split=0.25, verbose = 0, callbacks = [callback])
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
          epochs = 100,
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
      if propensity > 0:
        e_x = np.full((len(X[train_idx]),2), propensity)
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

    # shuffle data
    idx = np.random.RandomState(seed=(i+1)).permutation(pd.DataFrame(X).index)
    X = np.array(pd.DataFrame(X).reindex(idx))
    pseudo_outcome = np.array(pd.DataFrame(pseudo_outcome).reindex(idx))
    w_weigths = np.array(pd.DataFrame(w_weigths).reindex(idx))

    cv = KFold(n_splits=folds, shuffle = False)
    print("training for tau hat")
    for  k, (train_idx, test_idx) in enumerate(cv.split(X)):
      print(len(train_idx))
      tau_hat = tuner.hypermodel.build(best_hps)
      tau_hat.compile(optimizer=DPKerasAdamOptimizer(l2_norm_clip=4, noise_multiplier=noise_multiplier, num_microbatches=1, learning_rate=0.001),
                      loss= tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE),
                      metrics=[ATE])
      history_tau = tau_hat.fit(
          X[train_idx],
          pseudo_outcome[train_idx],
          sample_weight= w_weigths[train_idx],
          epochs = epochs,
          batch_size = batch_size,
          callbacks = tau_hat_callbacks,
          validation_data = (X[test_idx], pseudo_outcome[test_idx]),
          verbose = 1)
      tau_hat = tuner.hypermodel.build(best_hps)
      tau_hat.build(input_shape = (None,X.shape[1]))
      tau_hat.load_weights(checkpoint_filepath_taux)
      CATE = tau_hat.predict(x=X[test_idx], verbose = 0).reshape(len(X[test_idx]))
      print(f"Fold {k}: mean(tau_hat) = " + str(np.round(np.mean(CATE),2)) + ", sd(m_x) = " + str(np.round(np.std(CATE),3)))
      CATE_estimates = np.concatenate((CATE_estimates,CATE)) # store CATE's

    print("in-sample")
    tau_hat_in_sample = tuner.hypermodel.build(best_hps)
    history_tau = tau_hat_in_sample.fit(
        X,
        pseudo_outcome,
        sample_weight= w_weigths,
        epochs = epochs,
        batch_size = batch_size,
        callbacks = tau_hat_callbacks_insample,
        validation_split=0.25,
        verbose = 0)
    tau_hat_in_sample = tuner.hypermodel.build(best_hps)
    tau_hat_in_sample.build(input_shape = (None,X.shape[1]))
    tau_hat_in_sample.load_weights(checkpoint_filepath_taux)
    CATE_estimates_in_sample = tau_hat_in_sample.predict(x=X, verbose = 0).reshape(len(X))
    print(f"Fold {k}: mean(tau_hat_in_sample) = " + str(np.round(np.mean(CATE_estimates_in_sample),2)) + ", sd(m_x) = " + str(np.round(np.std(CATE_estimates_in_sample),3)))

    average_treatment_effect = np.append(average_treatment_effect, np.mean(CATE_estimates))
    print("ATE = " + str(np.round(np.mean(average_treatment_effect), 4)) + ", std(ATE) = " + str(np.round(np.std(average_treatment_effect), 3)))

  return average_treatment_effect, CATE_estimates, tau_hat, CATE_estimates_in_sample, tau_hat_in_sample

dataframe = pd.read_csv('data.csv', sep = ',', header = 0)
X = dataframe.iloc[:,1:121]
T = dataframe.iloc[:,121]
Y = dataframe.iloc[:,122]

X = np.array(X) # features
T = np.array(T) # treatment indicator
Y = np.array(Y) # revenue indicator

# noise_multipliers = [0.17596322, 0.200976, 0.218153, 0.4785, 0.8, 1.423, 6.3, 53] USED noise multipliers in STUDY
noise_multiplier = 0.17596322
average_treatment_effect, CATE_estimates, tau_hat, CATE_estimates_in_sample, tau_hat_in_sample = private_causal_neural_network(X = X, Y = Y, T = T, scaling = True, simulations = 1, epochs = 100,
                                                                                  max_epochs = 1, batch_size = 500, folds = 2, directory = "tuner_noise_500000",
                                                                                  noise_multiplier = noise_multiplier, propensity = 0)
np.savetxt("CATE_estimates_500000_tuning.csv", CATE_estimates, delimiter = ",")
np.savetxt("average_treatment_effect_500000_tuning.csv", average_treatment_effect, delimiter = ",")
tau_hat.save(filepath = "tauhat_500000_tuning")
%rm -rf tuner_noise_500000
%rm -rf m_x_1
%rm -rf tau_x1
