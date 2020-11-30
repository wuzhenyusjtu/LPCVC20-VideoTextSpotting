import torch
from torch.autograd import Variable
import misc

def validate_one_epoch(crnn, val_loader, criterion, converter, opt, logger=None, max_iter=100):
    if logger:
        logger.info('Start val')
    else:
        print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    crnn.eval()

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.cuda:
        image = image.cuda()

    image, text, length = Variable(image), Variable(text), Variable(length)

    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    loss_avg = misc.averager()

    max_iter = min(max_iter, len(val_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        misc.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        misc.loadData(text, t)
        misc.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        if logger:
            logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        else:
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    if logger:
        logger.info('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    else:
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train_one_epoch(crnn, train_loader, val_loader, criterion, optimizer, converter, epoch, opt, logger=None, callback=None):
    # Function for training one batch
    def train_one_batch(crnn, criterion, optimizer, train_iter):
        data = train_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        misc.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        misc.loadData(text, t)
        misc.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        return cost

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.cuda:
        image = image.cuda()

    image, text, length = Variable(image), Variable(text), Variable(length)

    train_iter = iter(train_loader)
    i = 0
    # loss averager
    loss_avg = misc.averager()

#     print("Start CRNN training.")
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = train_one_batch(crnn, criterion, optimizer, train_iter)
        loss_avg.add(cost)
        i += 1
        if i % opt.displayInterval == 0:
            if logger:
                logger.info('[%d/%d][%d/%d] Loss: %f' %
                            (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            else:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            validate_one_epoch(crnn, val_loader, criterion, converter, opt, logger=logger)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{}/netCRNN_{}_{}.pth'.format(opt.expr_dir, epoch, i))
            if logger:
                logger.info("Model saved to {}/netCRNN_{}_{}.pth".format(opt.expr_dir, epoch, i))
            else:
                print("Model saved to {}/netCRNN_{}_{}.pth".format(opt.expr_dir, epoch, i))
