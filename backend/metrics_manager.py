from math import log10

class metrics_manager(object):
	def __init__(self,name_list = ['miou']):
		self.metrics_dict={}
		for name in name_list:
			self.metrics_dict[name] = AverageMeter()
	def add_metrics(self,name_list):
		for name in name_list:
			self.metrics_dict[name] = AverageMeter()
	def update(self,name,value):
		self.metrics_dict[name].update(value)
	def reset(self):
		for name in self.metrics_dict:
			self.metrics_dict[name].reset()
	def return_metrics(self):
		return(self.metrics_dict)
	def summary(self):
		summery_dict={}
		for name in self.metrics_dict: 
			summery_dict[name]=self.metrics_dict[name].avg
		return(summery_dict)




def PSNR(mse, peak=1.):
	return 10 * log10((peak ** 2) / mse)


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

