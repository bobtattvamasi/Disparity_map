import time

# Декоратор для замера времени работы функций
def timer(method):
	def wraper(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		print(f"function {method.__name__} worked in {time.time() - ts} sec")
		return result
	return wraper
