import logging

def preprocessing(data, ctx):
	logging.info("Preprocessing for workflow smtest")
	logging.info("Input Daaaata type "+str(type(data))+" Dataaaa - "+str(data))
	if data:
		return data.tolist()
	else:
		return


def postprocessing(data, ctx):
	logging.info("Postprocessing for workflow smtest")
	if data:
		return data.tolist()
	else:
		return

