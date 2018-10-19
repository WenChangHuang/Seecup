import torch
class Lstm(torch.nn.Module):
	def __init__(self,opt):
		super(Lstm,self).__init__()
		self.embedding = torch.nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
		self.rnn = torch.nn.LSTM(
			input_size = opt.SENT_LEN,
			hidden_size = 512,
			num_layers = 3,
			batch_first = True,
			)	
		self.fc1 = torch.nn.Sequential(
                                torch.nn.Linear(51200,1024),
                                torch.nn.BatchNorm1d(1024),
                                torch.nn.ReLU(),
                                torch.nn.Linear(1024,opt.NUM_CLASS_1)
                                )

		self.fc2 = torch.nn.Sequential(
                                torch.nn.Linear(51200,1024),
                                torch.nn.BatchNorm1d(1024),
                                torch.nn.ReLU(),
                                torch.nn.Linear(1024,opt.NUM_CLASS_2)
                                )

		self.fc3 = torch.nn.Sequential(
				torch.nn.Linear(51200,1024),
				torch.nn.BatchNorm1d(1024),
				torch.nn.ReLU(),
				torch.nn.Linear(1024,opt.NUM_CLASS_3)
				)
	def forward(self,x):
		emb = self.embedding(x)
		output,(h_n,c_n) = self.rnn(emb)
		output = output.reshape([output.shape[0], output.shape[1]*output.shape[2]])
		output1 = self.fc1(output)
		output2 = self.fc2(output)
		output3 = self.fc3(output)
		return output1,output2,output3	
