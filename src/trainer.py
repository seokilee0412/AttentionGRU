from model import Attentionbased_GRU
from utils import FocalLoss
import torch.optim as optim
import torch
import os
import shutil
from tqdm import tqdm
import numpy as np

class Trainer:
	def __init__(self, model, train_loader, test_loader, cpu, cuda_devices=None):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		device = 'cuda' if torch.cuda.is_available and not cpu else 'cpu'
		self.device = device
		self.optimizer = optim.Adam(self.model.parameters())
		self.loss_func = FocalLoss()

	def training(self):
		self.model.train()
		loss, acc = self.iteration(self.train_loader, mode = 'train')
		return loss, acc

	def testing(self):
		self.model.eval()
		loss, acc = self.iteration(self.test_loader, mode = 'test')
		return loss, acc

	def iteration(self, data_loader, mode):
		losses = 0
		correct = 0
		data_cnt = 0
		bar = tqdm(data_loader, disable=False)
		# for x,y in data_loader:
		for x,y in bar:
			x = x.type(torch.float).to(self.device)
			y = y.to(self.device)

			output = self.model(x)

			loss = self.loss_func(output, y)
			probability, predict = torch.max(output, -1)
			correct += (predict==y).sum().item()
			losses += loss.item()
			data_cnt += 1


			if mode == 'train':
				self.model.zero_grad()
				loss.backward()
				self.optimizer.step()
			
		return np.round(losses/data_cnt, 4), np.round(correct/data_cnt, 4)

	def save_checkpoint(self, epoch, train_acc, test_acc, 
                                    test_loss, train_loss, is_best, save_dir):
		try:
			checkpoint = self.model.module.state_dict()
		except:
			checkpoint = self.model.state_dict()
			state = {'state_dict' : checkpoint,
					'test_accuarcy' : test_acc}
			torch.save(state, os.path.join(save_dir, 'model.ckpt'))
			if is_best:
				shutil.copy(os.path.join(save_dir, 'model.ckpt'), os.path.join(save_dir, 'model_best.ckpt'))

if __name__ == '__main__':
    model = Attentionbased_GRU(3,60,drop_prob=0.25,bidirectional=True)
    loss_func = FocalLoss()
