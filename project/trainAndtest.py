import torch
import os
import os.path
import pandas as pd

from models import *
from threshold import *
from torch.autograd import Variable

def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def set_lr(optimizer, up): 
    lr_now = 0
    max_lr = 0.01
    min_lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] - up

        lr_now = param_group['lr'] 

    print("lr = {}".format(lr_now))

    if lr_now <= min_lr:
        up = -0.0005
    elif lr_now >= max_lr:
        up = 0.0005

    return optimizer, up

# Have to add learning rate decay
def train(model, epoch, train_loader, optimizer, loss_fn):
    model.train()
    p_loss = 999999
    up = -0.0005
    # optimizer = lr_scheduler(optimizer, epoch)

    for id_batch, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target.float()).cuda()

        optimizer.zero_grad()
        output = model(data)

        # print(output)
        if model.__class__.__name__ == "GoogLeNet":
            output = output.logits
            
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        if id_batch % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, id_batch * len(data), len(train_loader.dataset),
                100. * id_batch / len(train_loader), loss.data))

            if p_loss < loss:
                print("Previos {}, new {}".format(p_loss, loss))
                optimizer, up = set_lr(optimizer, up)
            p_loss = loss

        # print(loss.data)
        # for param_group in optimizer.param_groups:
        # 	param_group['lr'] = param_group['lr'] * 10
        # if loss < 0.07:
        #     break
        if id_batch == 1000:
            break

def validate(model,val_loader,loss_func):
    model.eval()
    predictions = []
    true_labels = []
    thresholds = []
    val_out = []
    print ("Starting Validation")
    with torch.no_grad():
        for id_batch, (data, target) in enumerate(val_loader):

            data, target = data.cuda(async=True), target.cuda(async=True)
            data, target = Variable(data), Variable(target)
            outputs = model(data)
            
            predictions.append(torch.sigmoid(outputs))
            true_labels.append(target)
            val_out.append(outputs)

            if id_batch % 10 == 0:
                print("Done {}%".format(100 * id_batch * len(data)/float(len(val_loader.dataset))))

    tor_predictions = torch.cat(predictions)
    tor_true_labels = torch.cat(true_labels)
    tor_val_out = torch.cat(val_out)

    val_loss = loss_func(tor_val_out, tor_true_labels)


    print("Calling get_optimal_threshhold")
    predictions = tor_predictions.to(torch.device('cpu'))
    true_labels = tor_true_labels.to(torch.device('cpu'))
    
    threshold = best_f2_score(np.array(true_labels)[:len(predictions)], np.array(predictions))
    # threshold = [0.2]*17
    print("Got best threshold!")

    print ('--------------------------------------------------------')
    print ('[val_loss %.4f]' % val_loss)

    ff2 = fbeta(np.array(true_labels)[:len(predictions)], np.array(predictions)>threshold)

    print("Final fbeta score is " + str(ff2*100))
   
    return val_loss.data, threshold, ff2


def predict(test_loader, model, threshold, X_test, run_name):    
    model.eval()
    predictions = []
    true_labels = []

    dir_path = './out'
    mlb = X_test.getLabelEncoder()
    
    print("====== Starting Prediction ======")

    with torch.no_grad():
        for id_batch, (data, target) in enumerate(test_loader):
            data = data.cuda(async=True)
            data = Variable(data)
        
            pred = torch.sigmoid(model(data))
            predictions.append(pred.data.cpu().numpy())
            true_labels.append(target.data.cpu().numpy())

            if id_batch % 10 == 0:
                print("Done {}%".format(100 * id_batch * len(data)/float(len(test_loader.dataset))))
            
    
    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)

    print("====== Raw predictions done ========")

    predictions = predictions > threshold
    pred_path = os.path.join(dir_path, run_name + '-raw-pred-1'+'.csv')
    np.savetxt(pred_path, predictions, delimiter=";")
    
    result = pd.DataFrame({
        'image_name': X_test.name(),
        'tags': mlb.inverse_transform(predictions)
    })

    result['tags'] = result['tags'].apply(lambda tags: " ".join(tags))
    
    print("======= Final predictions done =======")
    
    ff2 = fbeta(np.array(true_labels)[:len(predictions)], np.array(predictions)>threshold)

    print("Final fbeta score is " + str(ff2*100))


# ************************Model Ensembline********************************
def ensembleResnetModelsPredictions(test_loader, model, threshold, X_test, run_name, models):
    main_predictions = None
    flag = True
    
    print("====== Starting Prediction Model Enesembling ======")

    true_labels = []


    for mt in models:
        mname = "./ensembler/{}.pth".format(mt)
        print("load model {}".format(mname))
        model = get_from_models(mt, 17, not(mname==None), mname)
        model.cuda()
        model.eval()
        predictions = []

        dir_path = './out'
        mlb = X_test.getLabelEncoder()
        

        with torch.no_grad():
            for id_batch, (data, target) in enumerate(test_loader):
                data = data.cuda(async=True)
                data = Variable(data)
            
                pred = torch.sigmoid(model(data))
                
                # pred = model(data)
                
                predictions.append(pred.data.cpu().numpy())
                if flag == True:
                    true_labels.append(target.data.cpu().numpy())

                if id_batch % 10 == 0:
                    print("Done {}%".format(100 * id_batch * len(data)/float(len(test_loader.dataset))))
                
        predictions = np.vstack(predictions)

        if flag == False:
            print("Main Prediction {}, prediction {}".format(main_predictions[0], predictions[0]))
            main_predictions = np.add(main_predictions , predictions)
        else:
            flag = False
            main_predictions = predictions

    predictions = main_predictions/len(models)     
    true_labels = np.vstack(true_labels)

    print("====== Raw predictions done ========")
    # predictions = predictions.numpy()
    predictions = predictions > threshold
    pred_path = os.path.join(dir_path, run_name + '-raw-pred-1'+'.csv')
    np.savetxt(pred_path, predictions, delimiter=";")
    
    result = pd.DataFrame({
        'image_name': X_test.name(),
        'tags': mlb.inverse_transform(predictions)
    })

    result['tags'] = result['tags'].apply(lambda tags: " ".join(tags))
    
    print("======= Final predictions done =======")
    
    result_path = os.path.join(dir_path, run_name + '-final-pred-'+'.csv')
    result.to_csv(result_path, index=False)

    print("Final predictions saved to {}".format(result_path))

    ff2 = fbeta(np.array(true_labels)[:len(predictions)], np.array(predictions)>threshold)

    print("Final fbeta score is " + str(ff2*100))
    