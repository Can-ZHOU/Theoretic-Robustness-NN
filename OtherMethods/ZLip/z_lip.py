""" Lipschitz (over)Estimation using zonotopes
	Github: https://github.com/revbucket/lipMIP
"""

import numpy as np
import math
from OtherMethods.ZLip.other_methods import OtherResult 
import OtherMethods.ZLip.utilities as utils 
from OtherMethods.ZLip.pre_activation_bounds import PreactivationBounds
from OtherMethods.ZLip.interval_analysis import AbstractNN
import OtherMethods.ZLip.bound_prop as bp 
from OtherMethods.ZLip.hyperbox import Hyperbox
from OtherMethods.ZLip.zonotope import Zonotope
from OtherMethods.ZLip.relu_nets import ReLUNet


class ZLip(OtherResult):
	def __init__(self, network, c_vector, domain, primal_norm):
		super(ZLip, self).__init__(network, c_vector, domain, primal_norm)

	def compute(self):
		# Fast lip is just interval bound propagation through backprop
		timer = utils.Timer()
		ap = bp.AbstractParams.basic_zono() 
		ann = bp.AbstractNN2(self.network) 

		self.grad_range = ann.get_both_bounds(ap, self.domain, self.c_vector)[1].output_range

		if self.primal_norm == 'linf': 
			value = self.grad_range.maximize_l1_norm_abs() 
		else:
			value = torch.max(self.grad_range.lbs.abs(), 
							  self.grad_range.ubs.abs()).max() 
		self.value = value
		self.compute_time = timer.stop()
		return value



# def test():
# 	layer_sizes = [4, 3, 2]
# 	network = ReLUNet(layer_sizes=layer_sizes, bias=True)
# 	x = [4.9,3.0,1.4,0.2]
# 	radius = 0.1
# 	c_vector = np.array([1.0, 0])
	
# 	result = ZipResult(network, c_vector, x, radius)
	
# 	return result