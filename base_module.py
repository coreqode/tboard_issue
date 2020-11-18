
def train(self):
    """ Complete Training Loop """
    with self.strategy.scope():
        self.define_model()
        self.define_dataset()
        self.define_metrics_meter()
        self.val_loss = tf.keras.metrics.Mean()  # for early stopping
    self.define_scheduler()
    self.define_optimizer()
    self.define_earlystopping()
    
    print("")
    print(f"Dataset Volume : {self.num_data}")
    print(f"Trainset Volume : {self.trainset_length}")
    print(f"Valset Volume : {self.valset_length}")
    print("")

    self.early_stopping.on_train_begin()

    for epoch in range(1, self.epochs + 1):

        #----------------------Training Loop -----------------------------#
        #-----------------------------------------------------------------#
        num_iterations = int(self.trainset_length // self.train_batch_size)
        bar = Bar(f"Ep : {epoch} | Training :", max=num_iterations)

        train_iterator = iter(self.train_gen)
        for self.step in range(num_iterations):
            if self.device == "colab_tpu":
                if (self.step == 0) and (epoch == 1):
                    losses = self.tpu_train_step_zero(train_iterator)
                    with self.strategy.scope():
                        self.define_loss_meter(losses)
                else:
                    self.tpu_train_step(train_iterator)

            else:
                data, model_inputs = next(train_iterator)
                if (self.step == 0) and (epoch == 1):
                    losses = self.train_step_zero(data, model_inputs)
                    with self.strategy.scope():
                        self.define_loss_meter(losses)
                else:
                    self.train_step(data, model_inputs)
                
            self._train_iter_dump_values(tf.constant(self.step, dtype=tf.int64))
            

            Bar.suffix = f"{self.step+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | {self.print_loss_metrics()}"
            bar.next()
        bar.finish()

        self._train_epoch_dump_values(tf.constant(epoch, dtype=tf.int64))

        #------------------------- Valdiation Loop -------------------------#
        #-------------------------------------------------------------------#
        num_iterations = int(self.valset_length // self.val_batch_size)
        bar = Bar(f"Ep : {epoch} | Validation :", max=num_iterations)

        val_iterator = iter(self.val_gen)
        for self.step in range(num_iterations):
            if self.device == "colab_tpu":
                self.tpu_val_step(val_iterator)
            else:
                inputs, targets = next(val_iterator)
                self.val_step(inputs, targets)
                
            self._val_iter_dump_values(tf.constant(self.step, dtype=tf.int64))

            Bar.suffix = f"{self.step+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | {self.print_loss_metrics()}"
            bar.next()
        bar.finish()
        
        self._val_epoch_dump_values(tf.constant(epoch, dtype=tf.int64))

        self.early_stopping.on_epoch_end(epoch, self.val_loss.result().numpy())
        self.initepoch()
        
@tf.function
def tpu_train_step_zero(self, iterator):
    losses = self.strategy.run(self.train_step_zero, args=(next(iterator)))
    return losses

@tf.function
def tpu_train_step(self, iterator):
    self.strategy.run(self.train_step, args=(next(iterator)))

@tf.function
def tpu_val_step(self, iterator):
    self.strategy.run(self.val_step, args=(next(iterator)))

@tf.function
def train_step_zero(self, data, model_inputs):
    with tf.GradientTape() as tape:
        predictions = self.model(model_inputs['inputs'], training=True)
        losses = self.loss_func(data, predictions)
        loss = sum(losses.values()) / self.train_batch_size

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        list(zip(gradients, self.model.trainable_variables))
    )
    self.update_metrics_meter(data, predictions)
    self.inputs = model_inputs['inputs']
    self.predictions = predictions['mask']
    self.targets = data['mask']
    self._train_dump_images(tf.constant(self.step, dtype=tf.int64))
    return losses

@tf.function
def train_step(self, data, model_inputs):
    """ Training for one step"""
    with tf.GradientTape() as tape:
        predictions = self.model(model_inputs['inputs'], training=True)
        losses = self.loss_func(data,  predictions)
        loss = sum(losses.values()) / self.train_batch_size
    
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        list(zip(gradients, self.model.trainable_variables))
    )
    self.update_loss_meter(losses, self.train_batch_size)
    self.update_metrics_meter(data, predictions)

@tf.function
def val_step(self, data, model_inputs):
    """ validation for one step"""
    predictions = self.model(model_inputs['inputs'])
    losses = self.loss_func(data, predictions)
    self.update_metrics_meter(data, predictions)
    self.update_loss_meter(losses, self.val_batch_size)
    loss = sum(losses.values())
    self.val_loss.update_state((loss * self.strategy.num_replicas_in_sync) / self.val_batch_size)

def tboard_dump_scalars(self, step, mode, style):
    with self.summary_writer.as_default():
        _ = [tf.summary.scalar(f"{name}/{mode}/{style}", loss.result(), step) for name,loss in self.loss_meter.items()]
        _ = [tf.summary.scalar(f"{name}/{mode}/{style}", metric.result(), step) for name,metric in self.metrics_meter.items()]
        if len(self.loss_meter)>1:
            tf.summary.scalar(f"loss_total/{mode}/{style}", sum([loss.result() for loss in self.loss_meter.values()]), step)

def tboard_dump_images(self, mode, step):
    with self.summary_writer.as_default():
        image  = self.tboard_image(self.inputs, self.targets, self.predictions)
        tf.summary.image(mode, image, step)

def tboard_image(self, inputs, targets, predictions, index=0):
    print(inputs.shape, targets.shape, predictions.shape)
    person = tf.cast(inputs[index:index + 1], tf.float32)
    gt_mask = tf.broadcast_to(tf.cast(targets[index: index + 1][..., :1], tf.float32), person.shape)
    predicted_mask = tf.cast(tf.expand_dims(B.argmax(predictions, axis = -1)[index:index + 1], axis = -1), tf.float32)
    predicted_mask = tf.broadcast_to(predicted_mask, person.shape)
    img_lists = [person, gt_mask, predicted_mask]
    image = tf.concat(img_lists, axis=0)
    return image

@tf.function
def _train_iter_dump_values(self, step): return self.tboard_dump_scalars(step, "train", "iter")

@tf.function
def _train_epoch_dump_values(self, step): return self.tboard_dump_scalars(step, "train", "epoch")

@tf.function
def _val_iter_dump_values(self, step): return self.tboard_dump_scalars(step, "val", "iter")

@tf.function
def _val_epoch_dump_values(self, step): return self.tboard_dump_scalars(step, "val", "epoch")

@tf.function
def _train_dump_images(self, step): return self.tboard_dump_images('train', step)

@tf.function
def _val_dump_images(self, step): return self.tboard_dump_images( 'val', step) 
