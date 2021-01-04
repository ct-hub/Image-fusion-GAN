################################################################################
# Visible-infrared image fusion.                                               #
# Implementation of the method by Zhao et al.:                                 #
# https://www.hindawi.com/journals/mpe/2020/3739040/                           #
#                                                                              #
# Tecnologico de Monterrey                                                     #
# MSc Computer Science                                                         #
# Jorge Francisco Ciprian Sanchez - A01373326                                  #
################################################################################

# Imports.
import configparser
import tensorflow as tf
from Functions.menu import *


# # GPU configuration
print("Configuring GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
  # Restrict TensorFlow to only use a given GPU
  try:
    print("GPUs [5]: ", gpus[5])
    tf.config.experimental.set_visible_devices(gpus[5], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Logical GPUs: ", logical_gpus)
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
print("... done.")

print("Reading configuration file...")
config = configparser.ConfigParser()
config.read('config.ini')
print("... done.")

main_menu(config)


# Executing program validating existing gpus.
# if(gpus):
#     try:
#         # Specify a GPU device
#         with tf.device(logical_gpus[0]):
#             main_menu()
#     except RuntimeError as e:
#       print(e)
# else:
#     main_menu()



# dateTimeObj = datetime.now()
# timestamp_str = dateTimeObj.strftime("%d-%b-%Y_(%H:%M:%S.%f)")
# print("TIMESTAMP: ", timestamp_str)
#
# # Creating directory with this name.
# dir_path = "./Checkpoints/" + timestamp_str + "/"
# os.makedirs(dir_path)

# Defining image paths.
# img_path_rgb = "./Train/VIS/*"
# img_path_ir = "./Train/IR/*"
#
# # img_path_rgb = "./Test/VIS/*"
# # img_path_ir = "./Test/IR/*"
#
# # Generating training and validation sets.
# rgb_images_train, rgb_images_val, ir_images_train, ir_images_val = \
# load_datasets(img_path_rgb,img_path_ir,v_split=True)
#
# save_model = False
# epochs = 4
# train(rgb_images_train, ir_images_train, save_model, epochs)

# Loading test dataset.
#rgb_images, ir_images = load_datasets(img_path_rgb,img_path_ir,v_split=False)







# Creating models.
# prev_gen_1 = create_g1()
# prev_gen_2 = create_g2()
# prev_disc_1 = create_d()
# prev_disc_2 = create_d()
#
# # Saving models.
# gen1_path = dir_path + "GEN1.h5"
# gen2_path = dir_path + "GEN2.h5"
# disc1_path = dir_path + "DISC1.h5"
# disc2_path = dir_path + "DISC2.h5"
#
# prev_gen_1.save_weights(gen1_path)
# prev_gen_2.save_weights(gen2_path)
# prev_disc_1.save_weights(disc1_path)
# prev_disc_2.save_weights(disc2_path)

# Loading models.
# gen_1 = create_g1()
# gen_2 = create_g2()
# disc_1 = create_d()
# disc_2 = create_d()

# gen_1.load_weights(gen1_path)
# gen_2.load_weights(gen2_path)
# disc_1.load_weights(disc1_path)
# disc_2.load_weights(disc2_path)

# cont = 0
#
# # Testing iterations over batches.
# for batch_rgb, batch_ir in zip(rgb_images_train, ir_images_train):
#     # Halting for testing purposes.
#     if(cont == 2): break
#     # Printing step (batch number).
#     print("Batch: ", cont)
#     # Displaying batch image sizes.
#     print("Size batch RGB: ",tf.shape(batch_rgb))
#     print("Size batch IR: ",tf.shape(batch_ir))
#     # # Displaying sample from batch RGB.
#     plt.figure(figsize=(10, 10))
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(batch_rgb[i+20].numpy().astype("uint8"))
#         #plt.imshow(image_batch_rgb[i+20])
#         plt.axis("off")
#     plt.show()
#     # Displaying sample from batch IR.
#     plt.figure(figsize=(10, 10))
#     for i in range(9):
#         ax = plt.subplot(3,3,i+1)
#         #plt.imshow(np.squeeze(image_batch_ir[i+20].numpy().astype("uint8")),cmap="gray")
#         plt.imshow(batch_ir[i+20].numpy().astype("uint8"))
#         #plt.imshow(image_batch_ir[i+20])
#         plt.axis("off")
#     plt.show()

    # Calculating the outputs of the different models.
    # Getting the output from Generator 1.
    # gen1_out = gen_1(batch_rgb)
    # # print("Output batch Gen1: ",tf.shape(gen1_out))
    # # aux_gen1_o = plt.imshow(gen1_out[0])
    # # plt.show()
    # # Generating the input for Generator 2.
    # # Concatenating visible and generated IR images to generate input to
    # # Generator 2.
    # in_gen2 = tf.concat([batch_rgb, gen1_out], 3)
    # #print("Shape in_gen2: ",tf.shape(in_gen2))
    # # Getting the output from Generator 2.
    # gen2_out = gen_2(in_gen2)
    # # print("Output batch Gen2: ",tf.shape(gen2_out))
    # # aux_gen2_o = plt.imshow(gen2_out[0])
    # # plt.show()
    # # Outputs of discriminators.
    # # Discriminator 1.
    # # Getting the output of Discriminator 1 for the generated fused image.
    # disc1_out_g2 = disc_1(gen2_out)
    # print("Output batch Disc1 fused: ",tf.shape(disc1_out_g2))
    # # Getting the output of Discriminator 1 for the RGB images.
    # disc1_out_rgb = disc_1(batch_rgb)
    # print("Output batch Disc1 RGB: ",tf.shape(disc1_out_rgb))
    # # Discriminator 2.
    # # Getting the output of Discriminator 2 for the generated fused image.
    # disc2_out_g2 = disc_2(gen2_out)
    # print("Output batch Disc2 fused: ",tf.shape(disc2_out_g2))
    # # Getting the output of Discriminator 2 for the generated IR image.
    # disc2_out_g1 = disc_2(gen1_out)
    # print("Output batch Disc2 fake IR: ",tf.shape(disc2_out_g1))
    # # Getting the output of Discriminator 2 for the real IR images.
    # disc2_out_ir = disc_2(batch_ir)
    # print("Output batch Disc2 real IR: ",tf.shape(disc2_out_ir))
    # # Calculating loss functions.
    # # Calculating loss for Generator 1.
    # gen1_loss = loss_g1(disc2_out_g1, gen1_out, batch_ir)
    # print("G1 cost: ",gen1_loss)
    # # Calculating loss for Generator 2.
    # gen2_loss = loss_g2(disc1_out_g2, disc2_out_g2, batch_ir, batch_rgb, gen2_out)
    # print("G2 cost: ",gen2_loss)
    # # Calculating loss for Discriminator 1.
    # disc1_loss = loss_d1(disc1_out_rgb, disc1_out_g2)
    # print("D1 cost: ",disc1_loss)
    # # Calculating loss for Discriminator 2.
    # disc2_loss = loss_d2(disc2_out_ir, disc2_out_g2, disc2_out_g1)
    # print("D2 cost: ",disc2_loss)
    # Updating batch counter.
    # cont += 1


#aug_train_ir =
#print("Augmented IR size: ",tf.data.experimental.cardinality(aug_train_ir).numpy())

# cont = 0
# # range(90)
# for aux in range(1):
#     cont+=1
#     image_batch_rgb = next(iter(rgb_images_train))
#     image_batch_ir = next(iter(ir_images_train))
#     # Displaying batch images.
#     print("Size batch RGB: ",tf.shape(image_batch_rgb))
#     print("Size batch IR: ",tf.shape(image_batch_ir))
#     cont = 0
#     plt.figure(figsize=(10, 10))
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(image_batch_rgb[i+20].numpy().astype("uint8"))
#         #plt.imshow(image_batch_rgb[i+20])
#         plt.axis("off")
#     plt.show()
#
#     plt.figure(figsize=(10, 10))
#     for i in range(9):
#         ax = plt.subplot(3,3,i+1)
#         #plt.imshow(np.squeeze(image_batch_ir[i+20].numpy().astype("uint8")),cmap="gray")
#         plt.imshow(image_batch_ir[i+20].numpy().astype("uint8"))
#         #plt.imshow(image_batch_ir[i+20])
#         plt.axis("off")
#     plt.show()
#     # Outputs of generators.
#     # Getting the output from Generator 1.
#     gen1_out = gen_1(image_batch_rgb)
#     print("Output batch Gen1: ",tf.shape(gen1_out))
#     aux_gen1_o = plt.imshow(gen1_out[0])
#     plt.show()
#     # Concatenating visible and generated IR images to generate input to
#     # Generator 2.
#     in_gen2 = tf.concat([image_batch_rgb, gen1_out], 3)
#     print("Shape in_gen2: ",tf.shape(in_gen2))
#     #in_gen2_rgb =  plt.imshow(in_gen2[0,:,:,0:3].numpy().astype("uint8"))
#     #plt.show()
#     #in_gen2_fir =  plt.imshow(in_gen2[0,:,:,3:6])
#     #plt.show()
#     # Getting the output from Generator 2.
#     gen2_out = gen_2(in_gen2)
#     print("Output batch Gen2: ",tf.shape(gen2_out))
#     aux_gen2_o = plt.imshow(gen2_out[0])
#     plt.show()
#     # Outputs of discriminators.
#     # Getting the output of Discriminator 1 for the generated fused image.
#     disc1_out_g2 = disc_1(gen2_out)
#     print("Output batch Disc1 fused: ",tf.shape(disc1_out_g2))
#     # Getting the output of Discriminator 1 for the RGB images.
#     disc1_out_rgb = disc_1(image_batch_rgb)
#     print("Output batch Disc1 RGB: ",tf.shape(disc1_out_rgb))
#     # Getting the output of Discriminator 2 for the generated fused image.
#     disc2_out_g2 = disc_2(gen2_out)
#     print("Output batch Disc2 fused: ",tf.shape(disc2_out_g2))
#     # Getting the output of Discriminator 2 for the generated IR image.
#     disc2_out_g1 = disc_2(gen1_out)
#     print("Output batch Disc2 fake IR: ",tf.shape(disc2_out_g1))
#     # Getting the output of Discriminator 2 for the real IR images.
#     disc2_out_ir = disc_2(image_batch_ir)
#     print("Output batch Disc2 real IR: ",tf.shape(disc2_out_ir))
#     # Calculating losses.
#     # Calculating loss for Generator 1.
#     gen1_loss = loss_g1(disc2_out_g1, gen1_out, image_batch_ir)
#     print("G1 cost: ",gen1_loss)
#     # Calculating loss for Generator 2.
#     gen2_loss = loss_g2(disc1_out_g2, disc2_out_g2, image_batch_ir, image_batch_rgb, gen2_out)
#     print("G2 cost: ",gen2_loss)
#     # Calculating loss for Discriminator 1.
#     disc1_loss = loss_d1(disc1_out_rgb, disc1_out_g2)
#     print("D1 cost: ",disc1_loss)
#     # Calculating loss for Discriminator 2.
#     disc2_loss = loss_d2(disc2_out_ir, disc2_out_g2, disc2_out_g1)
#     print("D2 cost: ",disc2_loss)
#     break
