import numpy as np
from classification_func import classification, classification_lr
import update_embed_mt2st_switch
import update_embed_mt2st_dimish
import update_embed_mt
from data_preprocessing import preprocess_dataset
from eval_func import sim_loss_cal, word_sim_dir,word_vec_file, read_word_vectors
from func_plot import plot2D_accuracy, plot2D_loss, plot2D2_loss, plot2D2_accuracy

def original_embedding(word_model):
  emb_table = []
  for i, word in enumerate(word_list):
    if word in word_model:
      emb_table.append(word_model[word])
    else:
      emb_table.append([0]*emb_dim)
  emb_table = np.array(emb_table)
  return emb_table

n_class, text_final, word_index, label, emb_dim, word_list, label_encoded, train_pad_encoded,test_pad_encoded,label_train_encoded,label_test_encoded,text_pad_encoded=preprocess_dataset()

mode = "mt2st_switch"
update_modes = "null"

if(mode=="mt"):
    update_modes = update_embed_mt
elif(mode=="mt2st_dimish"):
    update_modes = update_embed_mt2st_switch
elif(mode=="mt2st_switch"):
    update_modes = update_embed_mt2st_switch

# Pretrained embedding Glove-Twitter
import gensim.downloader as api
wv_emb_twitter_100 = api.load("glove-twitter-100")

label = "Original Methods"
print(label+"\n")
wv_twitter_100_emb = original_embedding(wv_emb_twitter_100)
# losses,train_accs = classification_lr(wv_twitter_100_emb,0.05, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded)
#
# epoches = [i for i in range(1, 26)]
# plot2D_loss(epoches, losses, "Epoches", "Loss",
#        f"Loss versus Epoches (Execution Mode - Original)",
#        f"Execution Mode - Original")
#
# plot2D_accuracy(epoches, train_accs, "Epoches", "Accuracy",
#        f"Accuracy versus Epoches (Execution Mode - Original)",
#        f"Execution Mode - Original")
#
# print("losses:\n",losses)
# print("train_accs:\n",train_accs)
#
print("UPDATE0 Methods\n")
original_wv_twitter_100_prediction = classification(wv_twitter_100_emb, n_class, 256)
if(update_modes!=update_embed_mt2st_switch):
    losses, train_accs = update_modes.update_embedding0(wv_twitter_100_emb, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded, test_pad_encoded,
                                                         label_test_encoded)
    epoches = [i for i in range(1, 26)]
    plot2D_loss(epoches, losses, "Epoches", "Loss",
           f"Loss versus Epoches (Execution Mode - {mode} Update Mode - Setting0)",
           f"Execution Mode - {mode} Update Mode - Setting0")

    plot2D_accuracy(epoches, train_accs, "Epoches", "Accuracy",
           f"Accuracy versus Epoches (Execution Mode - {mode} Update Mode - Setting0)",
           f"Execution Mode - {mode} Update Mode - Setting0")
else:
    losses_previous, losses_after, train_accs = update_modes.update_embedding0(wv_twitter_100_emb, n_class, emb_dim, word_list,
                                                        train_pad_encoded, label_train_encoded, test_pad_encoded,
                                                        label_test_encoded)
    epoches1 = [i for i in range(1,16)]
    epoches2 = [i for i in range(16,26)]

    plot2D2_loss(epoches1, losses_previous, epoches2, losses_after, "Epoches", "Loss",
            f"Loss versus Epoches (Mode - {mode} Update Mode - Setting0)", f"{mode} (before switch) - Setting0",
            f"{mode} (after switch) - Setting0")

    train_accs1=train_accs[:15]
    train_accs2=train_accs[15:]
    plot2D2_accuracy(epoches1, train_accs1, epoches2, train_accs2, "Epoches", "Accuracy",
            f"Accuracy versus Epoches (Mode - {mode} Update Mode - Setting0)", f"{mode} (before switch) - Setting0",
            f"{mode} (after switch) - Setting0")
#
# print("UPDATE1 Methods\n")
# if(update_modes!=update_embed_mt2st_switch):
#     losses, train_accs = update_modes.update_embedding1(wv_twitter_100_emb, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded, test_pad_encoded,
#                                                          label_test_encoded)
#     epoches = [i for i in range(1,26)]
#     plot2D_loss(epoches, losses, "Epoches", "Loss",
#            f"Loss versus Epoches (Execution Mode - {mode} Update Mode - Setting1)",
#            f"Execution Mode - {mode} Update Mode - Setting1")
#
#     plot2D_accuracy(epoches, train_accs, "Epoches", "Accuracy",
#            f"Accuracy versus Epoches (Execution Mode - {mode} Update Mode - Setting1)",
#            f"Execution Mode - {mode} Update Mode - Setting1")
# else:
#     losses_previous, losses_after, train_accs = update_modes.update_embedding1(wv_twitter_100_emb, n_class, emb_dim, word_list,
#                                                         train_pad_encoded, label_train_encoded, test_pad_encoded,
#                                                         label_test_encoded)
#
#     epoches1 = [i for i in range(1,16)]
#     epoches2 = [i for i in range(16,26)]
#
#     plot2D2_loss(epoches1, losses_previous, epoches2, losses_after, "Epoches", "Loss",
#             f"Loss versus Epoches (Mode - {mode} Update Mode - Setting1)", f"{mode} (before switch) - Setting1",
#             f"{mode} (after switch) - Setting1")
#
#     train_accs1=train_accs[:15]
#     train_accs2=train_accs[15:]
#     plot2D2_accuracy(epoches1, train_accs1, epoches2, train_accs2, "Epoches", "Accuracy",
#             f"Accuracy versus Epoches (Mode - {mode} Update Mode - Setting1)", f"{mode} (before switch) - Setting1",
#             f"{mode} (after switch) - Setting1")
#
print("UPDATE2 Methods\n")
if(update_modes!=update_embed_mt2st_switch):
    losses, train_accs = update_modes.update_embedding2(wv_twitter_100_emb, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded, test_pad_encoded,
                                                         label_test_encoded)
    epoches = [i for i in range(1,26)]
    plot2D_loss(epoches, losses, "Epoches", "Loss",
           f"Loss versus Epoches (Execution Mode - {mode} Update Mode - Setting2)",
           f"Execution Mode - {mode} Update Mode - Setting2")

    plot2D_accuracy(epoches, train_accs, "Epoches", "Accuracy",
           f"Accuracy versus Epoches (Execution Mode - {mode} Update Mode - Setting2)",
           f"Execution Mode - {mode} Update Mode - Setting2")
else:
    losses_previous, losses_after, train_accs = update_modes.update_embedding2(wv_twitter_100_emb, n_class, emb_dim, word_list,
                                                        train_pad_encoded, label_train_encoded, test_pad_encoded,
                                                        label_test_encoded)

    epoches1 = [i for i in range(1,16)]
    epoches2 = [i for i in range(16,26)]

    plot2D2_accuracy(epoches1, losses_previous, epoches2, losses_after, "Epoches", "Loss",
            f"Loss versus Epoches (Mode - {mode} Update Mode - Setting2)", f"{mode} (before switch) - Setting2",
            f"{mode} (after switch) - Setting2")

    train_accs1=train_accs[:15]
    train_accs2=train_accs[15:]
    plot2D2_accuracy(epoches1, train_accs1, epoches2, train_accs2, "Epoches", "Accuracy",
            f"Accuracy versus Epoches (Mode - {mode} Update Mode - Setting2)", f"{mode} (before switch) - Setting2",
            f"{mode} (after switch) - Setting2")


# print("UPDATE3 Methods\n")
# if(update_modes!=update_embed_mt2st_switch):
#     losses, train_accs = update_modes.update_embedding3(wv_twitter_100_emb, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded, test_pad_encoded,
#                                                          label_test_encoded)
#     epoches = [i for i in range(1,26)]
#     plot2D_loss(epoches, losses, "Epoches", "Loss",
#            f"Loss versus Epoches (Execution Mode - {mode} Update Mode - Setting3)",
#            f"Execution Mode - {mode} Update Mode - Setting3")
#
#     plot2D_accuracy(epoches, train_accs, "Epoches", "Accuracy",
#            f"Accuracy versus Epoches (Execution Mode - {mode} Update Mode - Setting3)",
#            f"Execution Mode - {mode} Update Mode - Setting3")
# else:
#     losses_previous, losses_after, train_accs = update_modes.update_embedding3(wv_twitter_100_emb, n_class, emb_dim, word_list,
#                                                         train_pad_encoded, label_train_encoded, test_pad_encoded,
#                                                         label_test_encoded)
#
#     epoches1 = [i for i in range(1,16)]
#     epoches2 = [i for i in range(16,26)]
#     plot2D2_accuracy(epoches1, losses_previous, epoches2, losses_after, "Epoches", "Loss",
#             f"Loss versus Epoches (Mode - {mode} Update Mode - Setting3)", f"{mode} (before switch) - Setting3",
#             f"{mode} (after switch) - Setting3")
#
#     train_accs1=train_accs[:15]
#     train_accs2=train_accs[15:]
#     plot2D2_accuracy(epoches1, train_accs1, epoches2, train_accs2, "Epoches", "Accuracy",
#             f"Accuracy versus Epoches (Mode - {mode} Update Mode - Setting3)", f"{mode} (before switch) - Setting3",
#             f"{mode} (after switch) - Setting3")