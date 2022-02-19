import numpy as np 
from .other_methods import OtherResult 
import OtherMethods.ZLip.utilities as utils 
from lipopt.old_code import utils as lutils
#import lipopt.old_code.utils as lutils 
#import lipopt.old_code as po 
import torch.nn as nn

class LipOpt(OtherResult):

	def __init__(self, network, c_vector, domain, primal_norm):
		assert utils.arraylike(c_vector)
		super(LipOpt, self).__init__(network, c_vector, domain, primal_norm)

	def compute():
		# Collect weights + biases 
		# Need to make a network with last linear layer being a cvec @weight 
		timer = utils.Timer()
		seq = self.network.net # Should be a sequential object 

		lastlin = seq[-1]
		new_lastlin = nn.Linear(lastlin.in_features, 1)
		new_lastlin.weight.data = (self.c_vector @ lastlin.weight.data).view(1, -1)
		new_lastlin.bias.data = self.c_vector @ lastlin.bias.data
		new_seq = nn.Sequential(*([_ for _ in seq[:-1]] + [new_lastlin]))
		weights, biases = lutils.weights_from_pytorch(new_seq) 
		layer_config = []
		for el in new_seq:
			if isinstance(el, nn.Linear):
				layer_config.append(el.in_features)
		layer_config.append(1)
		layer_config = tuple(layer_config)
		# Collect ub/lb from domain 
		lbs = domain.as_twocol()[:, 0].view(-1).detach().numpy()
		ubs = domain.as_twocol()[:, 1].view(-1).detach().numpy()



		fc = po.FullyConnected(weights, biases) 
		f = fc.grad_poly 
		g, lb, ub = fc.new_krivine_constr(p=1, lb=lb, ub=ub) 
		m = po.KrivineOptimizer.new_maxmize_serial(f, g, lb=lb, ub=ub, deg=len(weights), 
												   start_indices=fc.start_indices, 
												   layer_config=layer_config, 
												   solver='gurobi')
		self.value = m[0].objVal
		self.compute_time = timer.stop() 
		return self.value 