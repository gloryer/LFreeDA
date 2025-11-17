
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from spektral.layers import GINConv, GlobalAvgPool
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import   BatchNormalization, Activation
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam





class GIN0(Model):
    def __init__(self, channels, n_layers):
        super().__init__()
        self.conv1 = GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
        self.convs = []
        for _ in range(1, n_layers):
            self.convs.append(
                GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
            )
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(channels, activation="relu")
      
        

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        for conv in self.convs:
            x = conv([x, a])
        x = self.pool([x, i])
        x = self.dense1(x)
      
        return x

    
class AdvDA_GIN(object):
    def __init__(self,loader_source_train, 
                 loader_target_train,
                 loader_target_test, GIN, n_classes,
                 epochs=90):

        #source train and test dataset
        self.loader_source_tr = loader_source_train
        #self.loader_source_te= loader_source_test
        
        
        # Target train and test dataset
        
        self.loader_target_tr = loader_target_train
        self.loader_target_te= loader_target_test


        self.n_classes = n_classes
    
        
        #Latent dim for AE/VAE
        self.latent_dim = 128 #25
        
        
        self.generator = GIN 
        self.epochs = epochs 
        
        
        
        #Classifier
        
        class_input = Input(shape=(128,))
        #x = latent(class_input)
        x= Dense(128, activation = "relu")(class_input)
        class_output = Dense(self.n_classes, activation = "softmax")(x)
        
        self.classifier = Model(class_input, class_output, name="classifier")
        
        #Discriminator
        
        disc_input = Input(shape=(128,))
        #x = latent(disc_input)
        x = Dense(128, activation = "relu")(disc_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(128, activation = "relu")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        disc_output = Dense(2, activation = "softmax")(x)
        
        self.discriminator = Model(disc_input, disc_output, name="discriminator")


        
        
      
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        
        self.lr = 0.001 
        self.momentum = 0.9
        self.alpha = 0.0002


        
        
        self.task_optimizer=  Adam(1e-3)
        self.gen_optimizer = Adam(1e-3)
        self.disc_optimizer = Adam(1e-3)
        
        self.train_task_loss = tf.keras.metrics.Mean()
        self.train_disc_loss = tf.keras.metrics.Mean()
        self.train_gen_loss = tf.keras.metrics.Mean()
        self.train_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()

     
        
        self.test_target_task_loss = tf.keras.metrics.Mean()

        self.test_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()



        
        self.batch_size = 32

        


    def train_batch(self, x_source_train, y_source_train, x_target_train, y_target_train, epoch):
        
        
        source = np.tile([1,0], (y_source_train.shape[0], 1))
        target = np.tile([0,1], (y_target_train.shape[0], 1))
        
        target_fake = np.tile([0,1], (y_source_train.shape[0], 1))
        source_fake = np.tile([1,0], (y_target_train.shape[0], 1))
        
        with tf.GradientTape() as disc_tape:
            y_domain_pred_source = self.discriminator(self.generator(x_source_train, training=True), training=True)
            y_domain_pred_target = self.discriminator(self.generator(x_target_train, training=True), training=True)
            
            disc_loss = self.loss(source, y_domain_pred_source) +  self.loss(target, y_domain_pred_target)  
            
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)  
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
        self.train_disc_loss(disc_loss)

        with tf.GradientTape() as task_tape, tf.GradientTape() as gen_tape:
            
            #Forward pass
            y_class_pred_source = self.classifier(self.generator(x_source_train, training=True), training=True)
            y_class_pred_target = self.classifier(self.generator(x_target_train, training=True), training=True)
            y_domain_pred_source = self.discriminator(self.generator(x_source_train, training=True), training=True)
            y_domain_pred_target = self.discriminator(self.generator(x_target_train, training=True), training=True)
            
            
            task_loss = self.loss(y_target_train, y_class_pred_target) + 0.5*self.loss(y_source_train, y_class_pred_source)  
            adv_loss = self.loss(target_fake, y_domain_pred_source) +  self.loss(source_fake, y_domain_pred_target)   
            gen_loss = task_loss +  adv_loss*0.1
            
            #lp_grad = tape.gradient(lp_loss, self.predict_label.trainable_variables)
        
         # Compute gradients   
        task_grad = task_tape.gradient(task_loss, self.classifier.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        
        # Update weights 
        self.task_optimizer.apply_gradients(zip(task_grad, self.classifier.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables)) 
        
        
            

        self.train_task_loss(task_loss)
        self.train_task_accuracy(y_source_train, y_class_pred_source)
        self.train_target_task_accuracy(y_target_train, y_class_pred_target)
        self.train_gen_loss(gen_loss)
            
            


        return
    
    
    def test_batch(self, x_target_test, y_target_test):
       
       
        y_target_class_pred = self.classifier(self.generator(x_target_test, training=False), training=False)
        
            
        
        self.test_target_task_loss(y_target_test, y_target_class_pred)
        self.test_target_task_accuracy(y_target_test, y_target_class_pred)
        
        
        return 
    
    def evaluate(self, loader):
        output = []
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        step = 0
        while step < loader.steps_per_epoch:
            step += 1
            inputs, target = loader.__next__()
            pred = self.classifier(self.generator(inputs, training=False), training=False)
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )          
            output.append(outs)
            if step == loader.steps_per_epoch:
                output = np.array(output)
                return np.average(output[:, :-1], 0, weights=output[:, -1])

      
            


    
    def log_train(self):
        
        
        log_format = 'C_loss train: {:.4f}, Acc train source: {:.2f} , Acc train target: {:.2f}\n'+'D_loss train: {:.4f}, G_loss train: {:.4f}'

        message = log_format.format(
                 self.train_task_loss.result(),
                 self.train_task_accuracy.result()*100,
                 self.train_target_task_accuracy.result()*100,
                 self.train_disc_loss.result(),
                 self.train_gen_loss.result())
        

        self.reset_metrics('train')
        

        return message 
    
    def log_test(self):
        
        
        log_format = "C_loss test target: {:.4f}, Acc test target: {:.2f}"

        message = log_format.format(
                 self.test_target_task_loss.result(),
                 self.test_target_task_accuracy.result()*100)
                 
        

        self.reset_metrics('test')


        return message 
    
    def reset_metrics(self, target):

        if target == 'train':
            self.train_task_loss.reset_states()
            self.train_task_accuracy.reset_states()
            self.train_disc_loss.reset_states()
            self.train_gen_loss.reset_states()
            

        
        
        if target == 'test':
            self.test_target_task_loss.reset_states()
            self.test_target_task_accuracy.reset_states()
        

        return 
    
    def train(self):
        epoch = step = 0
        


        for (source_batch, source_labels), (target_batch, target_labels) in zip(self.loader_source_tr, self.loader_target_tr):
            step +=1 
            self.train_batch(source_batch, source_labels, target_batch, target_labels, epoch)
            if step == min(self.loader_source_tr.steps_per_epoch, self.loader_target_tr.steps_per_epoch):
                step = 0
                epoch +=1  
                if epoch % 10 ==0:
                    print('Epoch: {}'.format(epoch))
                    print(self.log_train())                 
                    results_te = self.evaluate(self.loader_target_te)
                    print("Test results - Loss: {:.3f} - Acc: {:.3f}".format(*results_te))
                    
                 

        return self.generator, self.classifier


class AdvDA_GIN_MD(object):
    def __init__(self,loader_source_train, 
                 loader_target_train,
                 loader_target_test, GIN, n_classes,
                 epochs=90):

        #source train and test dataset
        self.loader_source_tr = loader_source_train
        #self.loader_source_te= loader_source_test
        
        
        # Target train and test dataset
        
        self.loader_target_tr = loader_target_train
        self.loader_target_te= loader_target_test


        self.n_classes = n_classes
    
        
        #Latent dim for AE/VAE
        self.latent_dim = 256 #25
        
        
        self.generator = GIN 
        self.epochs = epochs 

        
        #Classifier
        
        class_input = Input(shape=(256,))
        #x = latent(class_input)
        x= Dense(256, activation = "relu")(class_input)
        class_output = Dense(self.n_classes, activation = "softmax")(x)
        
        self.classifier = Model(class_input, class_output, name="classifier")
        
        #Discriminator
        
        disc_input = Input(shape=(256,))
        #x = latent(disc_input)
        x = Dense(256, activation = "relu")(disc_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(256, activation = "relu")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        disc_output = Dense(2, activation = "softmax")(x)
        
        self.discriminator = Model(disc_input, disc_output, name="discriminator")

        
        
      
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        #self.loss_target = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

        
        self.lr = 0.001 
        self.momentum = 0.9
        self.alpha = 0.0002

        
        
        self.task_optimizer=  Adam(1e-3)
        self.gen_optimizer = Adam(1e-3)
        self.disc_optimizer = Adam(1e-3)
        
        self.train_task_loss = tf.keras.metrics.Mean()
        self.train_disc_loss = tf.keras.metrics.Mean()
        self.train_gen_loss = tf.keras.metrics.Mean()
        self.train_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()


        
        self.test_target_task_loss = tf.keras.metrics.Mean()
        #self.test_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.test_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        #self.test_target_f1_score =  tf.keras.metrics.F1Score(average="macro")


        
        self.batch_size = 32

        


    def train_batch(self, x_source_train, y_source_train, x_target_train, y_target_train, epoch):
        
        
        source = np.tile([1,0], (y_source_train.shape[0], 1))
        target = np.tile([0,1], (y_target_train.shape[0], 1))
        
        target_fake = np.tile([0,1], (y_source_train.shape[0], 1))
        source_fake = np.tile([1,0], (y_target_train.shape[0], 1))
        
        with tf.GradientTape() as disc_tape:
            y_domain_pred_source = self.discriminator(self.generator(x_source_train, training=True), training=True)
            y_domain_pred_target = self.discriminator(self.generator(x_target_train, training=True), training=True)
            
            disc_loss = self.loss(source, y_domain_pred_source) +  self.loss(target, y_domain_pred_target)  
            
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)  
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
        self.train_disc_loss(disc_loss)

        with tf.GradientTape() as task_tape, tf.GradientTape() as gen_tape:
            
            #Forward pass
            y_class_pred_source = self.classifier(self.generator(x_source_train, training=True), training=True)
            y_class_pred_target = self.classifier(self.generator(x_target_train, training=True), training=True)
            y_domain_pred_source = self.discriminator(self.generator(x_source_train, training=True), training=True)
            y_domain_pred_target = self.discriminator(self.generator(x_target_train, training=True), training=True)
            
            
            task_loss = self.loss(y_target_train, y_class_pred_target) + 0.5*self.loss(y_source_train, y_class_pred_source)  
            adv_loss = self.loss(target_fake, y_domain_pred_source) +  self.loss(source_fake, y_domain_pred_target)   
            gen_loss = task_loss  +  adv_loss*0.1
            
            #lp_grad = tape.gradient(lp_loss, self.predict_label.trainable_variables)
        
         # Compute gradients   
        task_grad = task_tape.gradient(task_loss, self.classifier.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) 
        
        # Update weights 
        self.task_optimizer.apply_gradients(zip(task_grad, self.classifier.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables)) 
        #self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
        
            

        self.train_task_loss(task_loss)
        self.train_task_accuracy(y_source_train, y_class_pred_source)
        self.train_target_task_accuracy(y_target_train, y_class_pred_target)
        #self.train_disc_loss(disc_loss)
        self.train_gen_loss(gen_loss)
            
            


        return
    
    
    def test_batch(self, x_target_test, y_target_test):
       
        # y_class_pred = self.classifier(self.generator(x_source_test, training=False), training=False)
        y_target_class_pred = self.classifier(self.generator(x_target_test, training=False), training=False)
        
            
        #self.test_task_loss.update_state(y_source_test, y_class_pred)
        self.test_target_task_loss(y_target_test, y_target_class_pred)
        #self.test_task_accuracy.update_state(y_source_test, y_class_pred)
        self.test_target_task_accuracy(y_target_test, y_target_class_pred)

        
        
        return 
    
    def evaluate(self, loader):
        output = []
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        step = 0
        while step < loader.steps_per_epoch:
            step += 1
            inputs, target = loader.__next__()
            pred = self.classifier(self.generator(inputs, training=False), training=False)
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
            output.append(outs)
            if step == loader.steps_per_epoch:
                output = np.array(output)
                return np.average(output[:, :-1], 0, weights=output[:, -1])

    
    def log_train(self):
        
        
        log_format = 'C_loss train: {:.4f}, Acc train source: {:.2f} , Acc train target: {:.2f}\n'+'D_loss train: {:.4f}, G_loss train: {:.4f}'

        message = log_format.format(
                 self.train_task_loss.result(),
                 self.train_task_accuracy.result()*100,
                 self.train_target_task_accuracy.result()*100,
                 self.train_disc_loss.result(),
                 self.train_gen_loss.result())
        

        self.reset_metrics('train')
        #self.reset_metrics('test')


        return message 
    
    def log_test(self):
        
        
        log_format = "C_loss test target: {:.4f}, Acc test target: {:.2f}"

        message = log_format.format(
                 #self.test_task_loss.result(),
                 #self.test_task_accuracy.result()*100,
                 self.test_target_task_loss.result(),
                 self.test_target_task_accuracy.result()*100)
                 #self.test_target_f1_score.result()*100)
        

        #self.reset_metrics('train')
        self.reset_metrics('test')


        return message 
    
    def reset_metrics(self, target):

        if target == 'train':
            self.train_task_loss.reset_states()
            self.train_task_accuracy.reset_states()
            self.train_disc_loss.reset_states()
            self.train_gen_loss.reset_states()
            
        #self.reset_metrics('test')
        
        
        if target == 'test':
            self.test_target_task_loss.reset_states()
            self.test_target_task_accuracy.reset_states()
        

        return 
    
    def train(self):
        epoch = step = 0



        for (source_batch, source_labels), (target_batch, target_labels) in zip(self.loader_source_tr, self.loader_target_tr):
            step +=1 
            self.train_batch(source_batch, source_labels, target_batch, target_labels, epoch)
            if step == min(self.loader_source_tr.steps_per_epoch, self.loader_target_tr.steps_per_epoch):
                step = 0
                epoch +=1  
                if epoch % 10 ==0:
                    print('Epoch: {}'.format(epoch))
                    print(self.log_train())                 
                    results_te = self.evaluate(self.loader_target_te)
                    print("Test results - Loss: {:.3f} - Acc: {:.3f}".format(*results_te))
                    


        return self.generator, self.classifier
                



        
                
            
            
     
            