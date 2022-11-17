from argparse import ArgumentParser
from trainer import Trainer
from preprocessing import make_dataloader
from model import Attentionbased_GRU
import os

def parse():
    args = ArgumentParser()

    args.add_argument('--lr', default = 3e-03, type=float)
    args.add_argument('--batch_size', default =16, type=int)
    args.add_argument('--epochs', default=30, type=int)

    args.add_argument('--cpu', default=True)#action='store_true')
    args.add_argument('--save_dir', default = None)

    arg = args.parse_args()
    return arg

def main():
    
    args = parse()
    
    print('Making Data Loader')
    train_loader, test_loader = make_dataloader(args.batch_size)
    print('Length of Train Loaer : ', len(train_loader))
    print('Length of Test Loader :', len(test_loader))

    print('Build Model')
    model = Attentionbased_GRU(input_size = 3, hidden_size = 512, drop_prob=0.25, bidirectional=True, num_layers=1)

    print('Build Trainer')
    trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, cpu = args.cpu)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0
    bad_per = 0
    train_accs = []
    test_accs = []
    print('Let"s Start Train!!')
    with open(os.path.join(save_dir, 'log'), 'a') as out:
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.training()
            test_loss, test_acc = trainer.testing()

            is_best = test_acc >= best_acc
            best_acc = max(test_acc, best_acc)
            
            if is_best:
                bad_per, best_epoch = 0, epoch
            else:
                bad_per += 1
            
            print('epoch: {}/{} | Train Loss : {} | Train Accuracy : {} | Test Loss : {} | Test Accuracy : {}'.format(
                epoch+1, args.epochs, train_loss, train_acc, test_loss, test_acc))

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            trainer.save_checkpoint(epoch, train_acc, test_acc, 
                                    test_loss, train_loss, is_best, save_dir)


            out.write('Epoch {}: train_loss={}, test_loss={}, train_acc={}, test_acc={}\n'.format(
                epoch+1, train_loss, test_loss, train_acc, test_acc))
            
            if bad_per > 5:
                print('test accuracy not improving for 5 epochs')
                break


if __name__ == '__main__':
    main()