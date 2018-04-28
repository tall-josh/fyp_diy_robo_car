import tensorflow as tf

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph("z_meta_save_test/ep_1_loss_16234.287.meta")
  new_saver.restore(sess, "z_meta_save_test/ep_1_loss_16234.287.ckpt")

  train = load_dataset(tain_path)
  test = load_dataset(tain_path)

  batch_size  = 50
  train_gen   = DataGenerator(batch_size=batch_size,
                    data_set=train,
                    image_dir=image_dir,
                    anno_dir=anno_dir,
                    num_bins=NUM_BINS)

 test_gen    = DataGenerator(batch_size=batch_size,
                    data_set=test,
                    image_dir=image_dir,
                    anno_dir=anno_dir,
                    num_bins=NUM_BINS)

 sample_gen  = DataGenerator(batch_size=10,
                    data_set=test[:10],
                    image_dir=image_dir,
                    anno_dir=anno_dir,
                    num_bins=NUM_BINS,
                    shuffle=False)

  for i in range(10):
    
