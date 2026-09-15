import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Layer, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch,12,12,2))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class MyDecay(LearningRateSchedule):

    def __init__(self, max_steps=1000, mu_0=0.01, alpha=10, beta=0.75):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.max_steps = float(max_steps)

    def __call__(self, step):
        p = step / self.max_steps
        return self.mu_0 / (1+self.alpha * p)**self.beta

class MaxDIrep(object):
    def __init__(self, x_source_train, y_source_train, 
                 x_target_train, y_target_train, 
                 x_source_test, y_source_test, 
                 x_target_test, y_target_test,
                 epochs=60):

        #source train and test dataset
        self.x_source_train = x_source_train
        self.y_source_train = y_source_train
        
        self.x_source_test = x_source_test
        self.y_source_test = y_source_test
        
        # Target train and test dataset
        
        self.x_target_train = x_target_train
        self.y_target_train = y_target_train

        self.x_target_test = x_target_test
        self.y_target_test = y_target_test


        self.n_classes = y_source_train.shape[1]

        # Use the source dataset shape for the generator input and outputs.
        self.input_shape = x_source_train.shape[1:]
        self.output_shape = y_source_train.shape[1:]

        self.latent_dim = (12,12,66)
        self.epochs = epochs

        self.generator = Sequential([
            Input(shape=(56,56,3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
        ], name="generator")

        self.classifier = Sequential([
            keras.Input(shape=(12, 12, 64)),
            Flatten(),
            Dense(256, activation = "relu"),
            Dense(2, activation = "softmax")
        ], name="classifier")

        self.discriminator = Sequential([
            keras.Input(shape=(12, 12, 64)),
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(2, activation='softmax')
        ], name="discriminator")

        encoder_inputs = keras.Input(shape=self.input_shape)
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)

        z_mean = Conv2D(filters=2, kernel_size=(3,3),strides=1,padding='valid',use_bias=False)(x)
        z_log_var = Conv2D(filters=2, kernel_size=(3,3),strides=1,padding='valid',use_bias=False)(x)
        z = Sampling()([z_mean, z_log_var])

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        self.decoder = Sequential([
            keras.Input(shape=(12, 12, 66), name="decoder_input"),
            # Project 66 channels to 64
            layers.Conv2D(64, kernel_size=1, activation="relu", padding="same", name="proj_conv"),
            # Increase spatial from 12→14 via transpose conv (valid padding)
            layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                name="deconv0"
            ),  # 12→14
            # Upsample 14→28
            layers.Conv2DTranspose(
                64,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name="deconv1"
            ),  # 14→28
            # Upsample 28→56
            layers.Conv2DTranspose(
                32,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name="deconv2"
            ),  # 28→56
            # Final reconstruction layer
            layers.Conv2DTranspose(
                3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="sigmoid",
                name="decoder_output"
            ),
        ], name="decoder")

        self.predict_label = Sequential([
            self.generator,
            self.classifier
        ])

        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()

        self.lr = 0.001
        self.momentum = 0.9
        self.alpha = 0.0002

        self.task_optimizer= Adam(0.001)
        self.gen_optimizer = Adam(0.001)
        self.disc_optimizer = Adam(0.001)
        self.enc_optimizer = Adam(0.001)
        self.dec_optimizer = Adam(0.001)

        self.train_task_loss = tf.keras.metrics.Mean()
        self.train_disc_loss = tf.keras.metrics.Mean()
        self.train_gen_loss = tf.keras.metrics.Mean()
        self.train_recon_loss = tf.keras.metrics.Mean()
        self.train_kl_loss = tf.keras.metrics.Mean()
        self.train_target_recon_loss = tf.keras.metrics.Mean()
        self.train_target_kl_loss = tf.keras.metrics.Mean()

        self.train_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_domain_accuracy = tf.keras.metrics.CategoricalAccuracy()

        self.test_task_loss = tf.keras.metrics.Mean()
        self.test_disc_loss = tf.keras.metrics.Mean()
        self.test_gen_loss = tf.keras.metrics.Mean()
        self.test_recon_loss = tf.keras.metrics.Mean()
        self.test_kl_loss = tf.keras.metrics.Mean()
        self.test_target_task_loss = tf.keras.metrics.Mean()
        self.test_target_recon_loss = tf.keras.metrics.Mean()
        self.test_target_kl_loss = tf.keras.metrics.Mean()

        self.test_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.test_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # 1) one Precision & Recall per class
        self.precision_per_class = [
            tf.keras.metrics.Precision(class_id=i, name=f"prec_{i}") for i in range(self.n_classes)
        ]
        self.recall_per_class = [
            tf.keras.metrics.Recall(class_id=i, name=f"rec_{i}") for i in range(self.n_classes)
        ]

        # 2) track support (count of true labels) for each class
        self.support_per_class = [
            tf.keras.metrics.Sum(name=f"support_{i}") for i in range(self.n_classes)
        ]

        self.batch_size = 32

    def KL(self, mean, log_var):
        loss_value = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        return tf.reduce_mean(tf.reduce_sum(loss_value, axis=1))

    def train_batch(self, x_source_train, y_source_train, x_target_train, epoch):
        source = np.tile([1,0], (x_source_train.shape[0], 1))
        target = np.tile([0,1], (x_target_train.shape[0], 1))
        domain_labels = np.concatenate([source, target], axis = 0)

        x_both = tf.concat([x_source_train, x_target_train], axis = 0)

        with tf.GradientTape() as disc_tape:
            y_domain_pred = self.discriminator(self.generator(x_both, training=True), training =True)
            disc_loss = self.loss(domain_labels, y_domain_pred)

        # Compute gradients
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Update weights
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

        self.train_disc_loss(disc_loss)

        with tf.GradientTape() as task_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            y_class_pred = self.classifier(self.generator(x_source_train, training=True), training =True)
            y_domain_pred = self.discriminator(self.generator(x_both, training=True), training =True)

            DIrep_source = self.generator(x_source_train)
            DIrep_target = self.generator(x_target_train)

            DDrep_mean_source, DDrep_log_var_source, DDrep_source = self.encoder(x_source_train, training = True)
            concat_samples_source = tf.concat([DIrep_source, DDrep_source], axis=3)

            DDrep_mean_target, DDrep_log_var_target, DDrep_target = self.encoder(x_target_train, training = True)
            concat_samples_target = tf.concat([DIrep_target, DDrep_target], axis=3)

            x_recon_source = self.decoder(concat_samples_source, training=True)
            x_recon_target = self.decoder(concat_samples_target, training=True)

            task_loss = self.loss(y_source_train, y_class_pred)
            recon_loss_source = self.mse(x_source_train, x_recon_source)
            recon_loss_target = self.mse(x_target_train, x_recon_target)
            kl_loss_source = self.KL(DDrep_mean_source, DDrep_log_var_source)
            kl_loss_target = self.KL(DDrep_mean_target, DDrep_log_var_target)

            recon_loss = recon_loss_source + recon_loss_target
            kl_loss =  kl_loss_source +  kl_loss_target

            total_loss = recon_loss + kl_loss*1/2000

            gen_loss = task_loss - disc_loss * 0.1  + recon_loss * 0.05

        # Compute gradients
        task_grad = task_tape.gradient(task_loss, self.classifier.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        enc_grad = enc_tape.gradient(total_loss, self.encoder.trainable_variables)
        dec_grad = dec_tape.gradient(total_loss, self.decoder.trainable_variables)

        # Update weights
        self.task_optimizer.apply_gradients(zip(task_grad, self.classifier.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.enc_optimizer.apply_gradients(zip(enc_grad, self.encoder.trainable_variables))
        self.dec_optimizer.apply_gradients(zip(dec_grad, self.decoder.trainable_variables))

        self.train_task_loss(task_loss)
        self.train_task_accuracy(y_source_train, y_class_pred)
        self.train_gen_loss(gen_loss)
        self.train_recon_loss(recon_loss_source)
        self.train_kl_loss(kl_loss_source)
        self.train_target_recon_loss(recon_loss_target)
        self.train_target_kl_loss(kl_loss_target)

        return

    def test_batch(self, x_source_test, y_source_test, x_target_test, y_target_test):
        # SOURCE domain (optional)
        if x_source_test is not None:
            # forward (no GradientTape; inference only)
            DIrep_source = self.generator(x_source_test, training=False)
            y_class_pred = self.classifier(DIrep_source, training=False)

            # classification metrics
            task_loss = self.loss(y_source_test, y_class_pred)
            self.test_task_loss(task_loss)
            self.test_task_accuracy(y_source_test, y_class_pred)

            # recon + KL (inference mode)
            DDrep_mean_source, DDrep_log_var_source, DDrep_source = self.encoder(x_source_test, training=False)
            concat_samples_source = tf.concat([DIrep_source, DDrep_source], axis=3)
            x_recon_source = self.decoder(concat_samples_source, training=False)

            recon_loss_source = self.mse(x_source_test, x_recon_source)
            kl_loss_source = self.KL(DDrep_mean_source, DDrep_log_var_source)

            self.test_recon_loss(recon_loss_source)
            self.test_kl_loss(kl_loss_source)

        # TARGET domain (optional)
        if x_target_test is not None:
            # forward (no GradientTape; inference only)
            DIrep_target = self.generator(x_target_test, training=False)
            y_target_class_pred = self.classifier(DIrep_target, training=False)

            # classification metrics
            target_task_loss = self.loss(y_target_test, y_target_class_pred)
            self.test_target_task_loss(target_task_loss)
            self.test_target_task_accuracy(y_target_test, y_target_class_pred)

            # recon + KL (inference mode)
            DDrep_mean_target, DDrep_log_var_target, DDrep_target = self.encoder(x_target_test, training=False)
            concat_samples_target = tf.concat([DIrep_target, DDrep_target], axis=3)
            x_recon_target = self.decoder(concat_samples_target, training=False)

            recon_loss_target = self.mse(x_target_test, x_recon_target)
            kl_loss_target = self.KL(DDrep_mean_target, DDrep_log_var_target)

            self.test_target_recon_loss(recon_loss_target)
            self.test_target_kl_loss(kl_loss_target)

            # 1) turn predictions into hard one-hot via argmax
            pred_ids = tf.argmax(y_target_class_pred, axis=1)
            pred_1hot = tf.one_hot(pred_ids, depth=self.n_classes, dtype=tf.float32)

            # 2) ensure y_true is one-hot float
            true_1hot = tf.cast(y_target_test, tf.float32)

            # 3) per-class P/R updates using hard one-hots
            for P, R in zip(self.precision_per_class, self.recall_per_class):
                P.update_state(true_1hot, pred_1hot)
                R.update_state(true_1hot, pred_1hot)

            # 4) support per class = number of true examples of that class
            true_ids = tf.argmax(true_1hot, axis=1)                    # shape (B,)
            for i, supp in enumerate(self.support_per_class):
                supp.update_state(tf.reduce_sum(tf.cast(true_ids == i, tf.float32)))

        return

    def log(self):
        eps = 1e-7
        # compute per-class F1 from the metric states (Macro-F1 only)
        f1s = []
        for P, R in zip(self.precision_per_class, self.recall_per_class):
            p = P.result()
            r = R.result()
            f1s.append(2.0 * p * r / (p + r + eps))

        macro_f1 = tf.reduce_mean(f1s)

        log_format = 'C_loss train: {:.4f}, Acc train : {:.2f}\n'+ \
            'D_loss train: {:.4f}, recon_loss_source:{:.4f}, kl_loss_source:{:.4f}, recon_loss_target {:.4f}, kl_loss_target {:.4f}\n'+ \
            'C_loss test source: {:.4f}, Acc test source: {:.2f}, recon_loss_source {:.4f}, kl_loss_source {:.4f}\n'+ \
            'C_loss test target: {:.4f}, Acc test target: {:.2f}, recon_loss_target {:.4f}, kl_loss_target {:.4f}\n' +\
            'Macro F1 test target: {:.2f}\n'
        message = log_format.format(
                 self.train_task_loss.result(),
                 self.train_task_accuracy.result()*100,
                 self.train_disc_loss.result(),
                 self.train_recon_loss.result(),
                 self.train_kl_loss.result(),
                 self.train_target_recon_loss.result(),
                 self.train_target_kl_loss.result(),
                 self.test_task_loss.result(),
                 self.test_task_accuracy.result()*100,
                 self.test_recon_loss.result(),
                 self.test_kl_loss.result(),
                 self.test_target_task_loss.result(),
                 self.test_target_task_accuracy.result()*100,
                 self.test_target_recon_loss.result(),
                 self.test_target_kl_loss.result(),
                 macro_f1.numpy()*100)

        self.reset_metrics('train')
        self.reset_metrics('test')

        return message

    def reset_metrics(self, target):
        if target == 'train':
            self.train_task_loss.reset_states()
            self.train_task_accuracy.reset_states()
            self.train_disc_loss.reset_states()
            self.train_recon_loss.reset_states()
            self.train_kl_loss.reset_states()
            self.train_target_recon_loss.reset_states()
            self.train_target_kl_loss.reset_states()

        if target == 'test':
            self.test_task_loss.reset_states()
            self.test_task_accuracy.reset_states()
            self.test_recon_loss.reset_states()
            self.test_kl_loss.reset_states()
            self.test_target_task_loss.reset_states()
            self.test_target_task_accuracy.reset_states()
            self.test_target_recon_loss.reset_states()
            self.test_target_kl_loss.reset_states()
            for m in self.precision_per_class + \
                     self.recall_per_class + \
                     self.support_per_class:
                m.reset_states()

        return

    def train(self):
        source_train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_source_train, self.y_source_train)
        ).shuffle(len(self.y_source_train)).batch(self.batch_size)

        # repeat target so it never runs out
        target_train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_target_train, self.y_target_train)
        ).shuffle(len(self.y_target_train)).batch(self.batch_size).repeat()

        source_test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_source_test, self.y_source_test)
        ).batch(self.batch_size)

        target_test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_target_test, self.y_target_test)
        ).batch(self.batch_size)

        steps_per_epoch = int(np.ceil(len(self.y_source_train) / self.batch_size))

        for epoch in range(self.epochs):
            source_iter = iter(source_train_dataset)
            target_iter = iter(target_train_dataset)

            for step in range(steps_per_epoch):
                source_images, source_labels = next(source_iter)
                target_images, _ = next(target_iter)
                self.train_batch(source_images, source_labels, target_images, epoch)

            # --- test phase (separate passes) ---
            for test_images, test_labels in source_test_dataset:
                self.test_batch(test_images, test_labels, None, None)

            for target_test_images, target_test_labels in target_test_dataset:
                self.test_batch(None, None, target_test_images, target_test_labels)

            print('Epoch: {}'.format(epoch + 1))
            print(self.log())

        return self.generator, self.classifier
