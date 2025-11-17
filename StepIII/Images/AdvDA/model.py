import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense,  Conv2D,  Flatten
from tensorflow.keras.layers import MaxPooling2D,  BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



class MyDecay(LearningRateSchedule):

    def __init__(self, max_steps=1000, mu_0=0.01, alpha=10, beta=0.75):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.max_steps = float(max_steps)

    def __call__(self, step):
        p = step / self.max_steps
        return self.mu_0 / (1+self.alpha * p)**self.beta
    

 
    


class AdvDA_CNN(object):
    def __init__(self, x_source_train, y_source_train, 
                 x_target_train, y_target_train, 
                 x_source_test, y_source_test, 
                 x_target_test, y_target_test, 
                 epochs=90):

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
        
       
        self.epochs = epochs 

        
        self.generator = Sequential([
            Input(shape=(56,56,3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten()
        ])
    

        self.classifier = Sequential([
            Dense(256, activation = "relu"),
            Dense(2, activation = "softmax")
        ])
        
        self.discriminator = Sequential([
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'), 
            Dense(2, activation='softmax')
        ])
        
        
 
        
      
        self.loss = tf.keras.losses.CategoricalCrossentropy()

       

    
        
        self.lr = 0.001 
        self.momentum = 0.9
        self.alpha = 0.0002

 
        self.task_optimizer= Adam(0.001)
        self.gen_optimizer = Adam(0.001)
        self.disc_optimizer = Adam(0.001)
       
        
        
        self.train_task_loss = tf.keras.metrics.Mean()
        self.train_target_task_loss = tf.keras.metrics.Mean()
        self.train_disc_loss = tf.keras.metrics.Mean()
        self.train_gen_loss = tf.keras.metrics.Mean()
        

        self.train_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_domain_accuracy = tf.keras.metrics.CategoricalAccuracy()


        self.test_task_loss = tf.keras.metrics.Mean()
        self.test_disc_loss = tf.keras.metrics.Mean()
        self.test_gen_loss = tf.keras.metrics.Mean()
        self.test_target_task_loss = tf.keras.metrics.Mean()
    

        self.test_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.test_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        

        
        self.batch_size = 32



    def train_batch(self, x_source_train, y_source_train, x_target_train, y_target_train, epoch):
        
        source = np.tile([1,0], (x_source_train.shape[0], 1))
        target = np.tile([0,1], (x_target_train.shape[0], 1))
        domain_labels = np.concatenate([source, target], axis = 0)
        class_labels = np.concatenate([y_source_train, y_target_train], axis = 0)
        
        
        x_both = tf.concat([x_source_train, x_target_train], axis = 0)
        
        
        with tf.GradientTape() as disc_tape:
            
            #Forward pass
            y_domain_pred = self.discriminator(self.generator(x_both, training=True), training =True)
            disc_loss = self.loss(domain_labels, y_domain_pred)  
          
        
         # Compute gradients   
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) 
        
        # Update weights 
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
    
        self.train_disc_loss(disc_loss)
       

        with tf.GradientTape() as task_tape, tf.GradientTape() as gen_tape:
            
            #Forward pass
            y_class_pred = self.classifier(self.generator(x_source_train, training=True), training =True)
            y_class_pred_target =  self.classifier(self.generator(x_target_train, training=True), training =True)
            y_domain_pred = self.discriminator(self.generator(x_both, training=True), training =True)
        
            task_loss_source = self.loss(y_source_train, y_class_pred) 
            task_loss_target = self.loss(y_target_train, y_class_pred_target)
            
            task_loss = task_loss_source + task_loss_target
            disc_loss = self.loss(domain_labels, y_domain_pred)  
            
        
            gen_loss = task_loss - disc_loss * 0.1
           
            
            
        
        
         # Compute gradients   
        task_grad = task_tape.gradient(task_loss, self.classifier.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) 
        
        
        # Update weights 
        self.task_optimizer.apply_gradients(zip(task_grad, self.classifier.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables)) 
        #self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
        
        self.train_task_loss(task_loss_source)
        self.train_task_accuracy(y_source_train, y_class_pred)
        
        self.train_target_task_loss(task_loss_target)
        self.train_target_task_accuracy(y_target_train, y_class_pred_target)
        
        self.train_gen_loss(gen_loss)
       

        return #self.generator, self.classifier
    
    def test_batch(self, x_source_test, y_source_test, x_target_test, y_target_test):
        
       
        
        
        with tf.GradientTape() as tape:
            
            DIrep_source = self.generator(x_source_test, training=False)
            DIrep_target = self.generator(x_target_test, training=False)
            
            y_class_pred = self.classifier(DIrep_source, training=False)
            y_target_class_pred = self.classifier(DIrep_target, training=False)
            
    
            task_loss = self.loss(y_source_test, y_class_pred)
            target_task_loss = self.loss(y_target_test, y_target_class_pred)
           
            

        self.test_task_loss(task_loss)
        self.test_task_accuracy(y_source_test, y_class_pred)
       


        self.test_target_task_loss(target_task_loss)
        self.test_target_task_accuracy(y_target_test, y_target_class_pred)

        return
    
    def test(self, x_source_test, y_source_test, x_target_test, y_target_test):
        
        with tf.GradientTape() as tape:
            y_class_pred = self.predict_label(x_source_test, training=False)
            y_target_class_pred = self.predict_label(x_target_test, training=False)
            
    
            task_loss = self.loss(y_source_test, y_class_pred)
            target_task_loss = self.loss(y_target_test, y_target_class_pred)
           
            

        self.test_task_loss(task_loss)
        self.test_task_accuracy(y_source_test, y_class_pred)
        
        
   

        self.test_target_task_loss(target_task_loss)
        self.test_target_task_accuracy(y_target_test, y_target_class_pred)

        return
    
    def log(self):
        
        
        log_format = 'c_loss source train: {:.4f}, acc source train : {:.2f}\n'+ \
            'c_loss target train: {:.4f}, acc target train : {:.2f}\n'+ \
            'D_loss train: {:.4f}\n'+ \
            'C_loss test source: {:.4f}, Acc test source: {:.2f}\n'+ \
            'C_loss test target: {:.4f}, Acc test target: {:.2f}\n'

        message = log_format.format(
                 self.train_task_loss.result(),
                 self.train_task_accuracy.result()*100,
                 self.train_target_task_loss.result(),
                 self.train_target_task_accuracy.result()*100,
                 self.train_disc_loss.result(),
                 self.test_task_loss.result(),
                 self.test_task_accuracy.result()*100,
                 self.test_target_task_loss.result(),
                 self.test_target_task_accuracy.result()*100)
        

        self.reset_metrics('train')
        self.reset_metrics('test')


        return message 
    
    def reset_metrics(self, target):

        if target == 'train':
            self.train_task_loss.reset_states()
            self.train_task_accuracy.reset_states()
            self.train_target_task_loss.reset_states()
            self.train_target_task_accuracy.reset_states()
            self.train_disc_loss.reset_states()
        
        if target == 'test':
            self.test_task_loss.reset_states()
            self.test_task_accuracy.reset_states()
            self.test_target_task_loss.reset_states()
            self.test_target_task_accuracy.reset_states()



        return
    
    
    def train(self):
        
        source_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_source_train, self.y_source_train)).shuffle(len(self.y_source_train)).batch(self.batch_size)
        target_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_target_train, self.y_target_train)).shuffle(len(self.y_target_train)).batch(self.batch_size)
        
        source_test_dataset = tf.data.Dataset.from_tensor_slices((self.x_source_test, self.y_source_test)).batch(self.batch_size)
        target_test_dataset = tf.data.Dataset.from_tensor_slices((self.x_target_test, self.y_target_test)).batch(self.batch_size)
        

        
        
        for epoch in range(self.epochs):
            
            batches = 0 
            
            for (source_images, source_labels), (target_images, target_labels) in zip(source_train_dataset, target_train_dataset):
                self.train_batch(source_images, source_labels, target_images, target_labels, epoch)
            

            for (test_images, test_labels), (target_test_images, target_test_labels) in zip(source_test_dataset, target_test_dataset):
                self.test_batch(test_images, test_labels, target_test_images, target_test_labels)
       
            print('Epoch: {}'.format(epoch + 1))
            print(self.log())
            
        return self.generator, self.classifier

 
    


class AdvDA_CNN_MD(object):
    def __init__(self, x_source_train, y_source_train, 
                 x_target_train, y_target_train, 
                 x_target_test, y_target_test, 
                 epochs=90):

        #source train and test dataset
        self.x_source_train = x_source_train
        self.y_source_train = y_source_train
        

        
        # Target train and test dataset
        
        self.x_target_train = x_target_train
        self.y_target_train = y_target_train

        self.x_target_test = x_target_test
        self.y_target_test = y_target_test


        self.n_classes = y_source_train.shape[1]
        
         
        # Use the source dataset shape for the generator input and outputs.
        self.input_shape = x_source_train.shape[1:]
        self.output_shape = y_source_train.shape[1:]
        
     
        self.epochs = epochs
        
       
      
        self.generator = Sequential([
            Input(shape=(56,56,3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2))
        ])
    

        self.classifier = Sequential([
            
            Input(shape=(12, 12, 64)),
            Flatten(),
            Dense(256, activation = "relu"),
            Dropout(0.5),  
            Dense(self.n_classes, activation = "softmax")
        ])
        
        self.discriminator = Sequential([
            Input(shape=(12, 12, 64)),
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'), 
            Dense(2, activation='softmax')
        ])
        

      
    
        
      
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        # target loss with label smoothing
        self.loss_target = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        
        self.target_loss_weight_max = 1.0
        self.ramp_up_epochs       = 10

       

    
        
        self.lr = 0.001 
        self.momentum = 0.9
        self.alpha = 0.0002

   
        self.task_optimizer= Adam(0.001)
        self.gen_optimizer = Adam(0.001)
        self.disc_optimizer = Adam(0.001)
       
        
        
        self.train_task_loss = tf.keras.metrics.Mean()
        self.train_disc_loss = tf.keras.metrics.Mean()
        self.train_gen_loss = tf.keras.metrics.Mean()
        

        self.train_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_domain_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        
        
        # --- new train‐time target metrics ---
        self.train_target_task_loss     = tf.keras.metrics.Mean()
        self.train_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()



        #self.test_task_loss = tf.keras.metrics.Mean()
        self.test_disc_loss = tf.keras.metrics.Mean()
        self.test_gen_loss = tf.keras.metrics.Mean()
        self.test_target_task_loss = tf.keras.metrics.Mean()
    

        #self.test_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
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
        
        self.test_target_f1 = tf.keras.metrics.Mean()
        
        self.batch_size = 32



    def train_batch(self, x_source_train, y_source_train, x_target_train, y_target_train, epoch):
        
        source = np.tile([1,0], (x_source_train.shape[0], 1))
        target = np.tile([0,1], (x_target_train.shape[0], 1))
        domain_labels = np.concatenate([source, target], axis = 0)
       
        x_both = tf.concat([x_source_train, x_target_train], axis = 0)
        
        
        with tf.GradientTape() as disc_tape:
            
            #Forward pass
            y_domain_pred = self.discriminator(self.generator(x_both, training=True), training =True)
            disc_loss = self.loss(domain_labels, y_domain_pred)  
          
        
         # Compute gradients   
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) 
        
        # Update weights 
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
    
        self.train_disc_loss(disc_loss)
       

        with tf.GradientTape() as task_tape, tf.GradientTape() as gen_tape:
            
            #Forward pass
            y_class_pred = self.classifier(self.generator(x_source_train, training=True), training =True)
            y_class_pred_target =  self.classifier(self.generator(x_target_train, training=True), training =True)
            y_domain_pred = self.discriminator(self.generator(x_both, training=True), training =True)
        
            task_loss_source = self.loss(y_source_train, y_class_pred) 
            task_loss_target = self.loss_target(y_target_train, y_class_pred_target)
    
            
            task_loss = task_loss_source*0.5 + task_loss_target 
            
            
            
            
            disc_loss = self.loss(domain_labels, y_domain_pred)  
            
        
            gen_loss = task_loss - disc_loss * 0.1
           
            
            
        
        
         # Compute gradients   
        task_grad = task_tape.gradient(task_loss, self.classifier.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        
        
        # Update weights 
        self.task_optimizer.apply_gradients(zip(task_grad, self.classifier.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables)) 
        
        
        self.train_task_loss(task_loss_source)
        self.train_task_accuracy(y_source_train, y_class_pred)
        
        self.train_target_task_loss(task_loss_target)
        self.train_target_task_accuracy(y_target_train, y_class_pred_target)
        
        self.train_gen_loss(gen_loss)
       

        return 
    
    def test_batch(self, x_source_test, y_source_test, x_target_test, y_target_test):
        
        
    
        with tf.GradientTape() as tape:
            
            DIrep_source = self.generator(x_source_test, training=False)
            DIrep_target = self.generator(x_target_test, training=False)
            
            y_class_pred = self.classifier(DIrep_source, training=False)
            y_target_class_pred = self.classifier(DIrep_target, training=False)
            
    
            task_loss = self.loss(y_source_test, y_class_pred)
            target_task_loss = self.loss(y_target_test, y_target_class_pred)
           
            

        self.test_task_loss(task_loss)
        self.test_task_accuracy(y_source_test, y_class_pred)
       


        self.test_target_task_loss(target_task_loss)
        self.test_target_task_accuracy(y_target_test, y_target_class_pred)
        
        for P, R in zip(self.precision_per_class, self.recall_per_class):
            P.update_state(y_target_test, y_target_class_pred)
            R.update_state(y_target_test, y_target_class_pred)

        # update support counts
        for i, supp in enumerate(self.support_per_class):
            # mask of batch entries equal to class i
            mask = tf.cast(tf.equal(y_target_test, i), tf.float32)
            supp.update_state(mask)

        return
    
    def evaluate(self):
        

 
        DIrep_target = self.generator(self.x_target_test, training=False)

        y_target_class_pred = self.classifier(DIrep_target, training=False)

    
        target_task_loss = self.loss(self.y_target_test, y_target_class_pred)
        
            
   

        self.test_target_task_loss(target_task_loss)
        self.test_target_task_accuracy(self.y_target_test, y_target_class_pred)
        
        for P, R in zip(self.precision_per_class, self.recall_per_class):
            P.update_state(self.y_target_test, y_target_class_pred)
            R.update_state(self.y_target_test, y_target_class_pred)
        
        labels = tf.argmax(self.y_target_test, axis=1)

        # update support counts
        for i, supp in enumerate(self.support_per_class):
            # mask of batch entries equal to class i
            mask = tf.cast(tf.equal(labels, i), tf.float32)
            supp.update_state(mask)
            
        # — compute weighted F1 and update the new metric
        eps = 1e-7
        f1s = [
            2 * P.result() * R.result() / (P.result() + R.result() + eps)
            for P, R in zip(self.precision_per_class, self.recall_per_class)
        ]
        supports = [supp.result() for supp in self.support_per_class]
        total_support = tf.reduce_sum(supports)
        weighted_f1 = tf.reduce_sum([f * s for f, s in zip(f1s, supports)]) / (total_support + eps)

        self.test_target_f1.update_state(weighted_f1)

    
    def log(self):
        
    
        
        
        log_format = 'c_loss source train: {:.4f}, acc source train : {:.2f}\n'+ \
            'c_loss target train: {:.4f}, acc target train : {:.2f}\n'+ \
            'D_loss train: {:.4f}\n'+ \
            'C_loss test target: {:.4f}, Acc test target: {:.2f}, F1 test target: {:.2f}\n'

        message = log_format.format(
                 self.train_task_loss.result(),
                 self.train_task_accuracy.result()*100,
                 self.train_target_task_loss.result(),
                 self.train_target_task_accuracy.result()*100,
                 self.train_disc_loss.result(),
                 self.test_target_task_loss.result(),
                 self.test_target_task_accuracy.result()*100,
                 self.test_target_f1.result()*100)
        

        self.reset_metrics('train')
        self.reset_metrics('test')


        return message 
    
    def reset_metrics(self, target):

        if target == 'train':
            self.train_task_loss.reset_states()
            self.train_task_accuracy.reset_states()
            self.train_disc_loss.reset_states()
            self.train_target_task_loss.reset_states()
            self.train_target_task_accuracy.reset_states()
        
        if target == 'test':
            self.test_target_task_loss.reset_states()
            self.test_target_task_accuracy.reset_states()
            self.test_target_f1.reset_states()
            for m in self.precision_per_class + \
                     self.recall_per_class + \
                     self.support_per_class:
                m.reset_states()



        return
    
    
    def train(self):
        
        source_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_source_train, self.y_source_train)).shuffle(len(self.y_source_train)).batch(self.batch_size)
        target_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_target_train, self.y_target_train)).shuffle(len(self.y_target_train)).batch(self.batch_size).repeat() 
        
        
        
        for epoch in range(self.epochs):
            
            batches = 0 
            
            for (source_images, source_labels), (target_images, target_labels) in zip(source_train_dataset, target_train_dataset):
                self.train_batch(source_images, source_labels, target_images, target_labels, epoch)
            
            
            
            self.evaluate()
            print('Epoch: {}'.format(epoch + 1))
            print(self.log())
            
        return self.generator, self.classifier

    
    
    
    
    