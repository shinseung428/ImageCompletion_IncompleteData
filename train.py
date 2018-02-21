import tensorflow as tf
from config import *
from network import *

import cv2

def train(args, sess, model):
    #optimizers
    g_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_G").minimize(model.g_loss, var_list=model.g_vars)
    d_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_D").minimize(model.d_loss, var_list=model.d_vars)

    # clipping weights
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in model.d_vars]

    epoch = 0
    step = 0
    global_step = 0

    #saver
    saver = tf.train.Saver()        
    if args.continue_training:
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print "Loaded model file from " + ckpt_name
        epoch = int(ckpt_name.split('-')[-1])
        tf.local_variables_initializer().run()
    else:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #summary init
    all_summary = tf.summary.merge([model.Y_r_sum,
                                    model.X_g_sum,
                                    model.Y_g_sum, 
                                    model.d_loss_sum,
                                    model.g_loss_sum])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)

    #training starts here
    while epoch < args.epochs:

        #Update Discriminator
        summary, d_loss, _ = sess.run([all_summary, model.d_loss, d_optimizer])
        writer.add_summary(summary, global_step)

        #Update Generator
        summary, g_loss, _ = sess.run([all_summary, model.g_loss, g_optimizer])
        writer.add_summary(summary, global_step)
        #Update Generator Again
        summary, g_loss, _ = sess.run([all_summary, model.g_loss, g_optimizer])
        writer.add_summary(summary, global_step)


        print "Epoch [%d] Step [%d] G Loss: [%.4f] D Loss: [%.4f]" % (epoch, step, g_loss, d_loss)

        if step*args.batch_size >= model.data_count:
            saver.save(sess, args.checkpoints_path + "/model", global_step=epoch)

            imgs = sess.run([model.Y_r,model.X_g])
            
            ##post processing test
            # test = sess.run([model.X_g,model.masks])

            # img = cv2.cvtColor(test[0][0], cv2.COLOR_BGR2RGB)
            # img = (img + 1) * 127.5
            # img = img.astype('uint8')
            
            # mask = cv2.cvtColor(test[1][0], cv2.COLOR_BGR2RGB)
            # mask = 255 - ((mask+1)*127.5)
            # mask = mask.astype('uint8')

            
            # dst = cv2.inpaint(img,mask[:,:,0],1,cv2.INPAINT_TELEA)
            # # dst = img
            # dst = dst.astype('float32')
            # mask = mask.astype('float32')

            # cv2.imshow("orig",img)
            # cv2.imshow("mask",mask/255)
            # cv2.imshow("test",dst/255)
            # cv2.waitKey()
            

            #saving image tile
            img_tile(epoch, args, imgs[0], name="input")
            img_tile(epoch, args, imgs[1], name="completed")

            step = 0
            epoch += 1 

        step += 1
        global_step += 1



    coord.request_stop()
    coord.join(threads)
    sess.close()            
    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        model = network(args)
        args.images_path = os.path.join(args.images_path, args.measurement)
        args.graph_path = os.path.join(args.graph_path, args.measurement)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.measurement)

        #create graph, images, and checkpoints folder if they don't exist
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)
        if not os.path.exists(args.images_path):
            os.makedirs(args.images_path)

        print 'Start Training...'
        train(args, sess, model)

main(args)
