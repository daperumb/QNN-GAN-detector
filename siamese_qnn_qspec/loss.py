import torch
import torch.nn as nn

def siamese_loss(batch1, batch2):

    batch = torch.cat((batch1, batch2))
    dist_map = torch.zeros(batch.size()[0], batch.size()[0], requires_grad=False)

    for i in range(batch.size()[0]):
        for j in range(batch.size()[0]):
            dist_map[i][j] = torch.dist(batch[i], batch[j])**2

    dist_map = -dist_map
    loss_i = torch.zeros(batch.size()[0])
    row_num = 0
    for i in dist_map:

        temp1, temp2, temp3 = i.split([row_num, 1, i.size()[0] - row_num - 1])
        dist_row = torch.cat((temp1, temp3))
        dist_row = torch.as_tensor(dist_row)

        row_softmax = nn.functional.softmax(dist_row, dim=0)
        if row_num < batch1.size()[0]:
            row_softmax1, row_softmax2 = row_softmax.split([batch1.size()[0]-1, batch2.size()[0]])
            loss_i[row_num] = -torch.log(torch.sum(row_softmax1))
        if row_num >= batch1.size()[0]:
            row_softmax1, row_softmax2 = row_softmax.split([batch1.size()[0], batch2.size()[0]-1])
            loss_i[row_num] = -torch.log(torch.sum(row_softmax2))

        row_num += 1

    loss = torch.sum(loss_i)
    return loss
