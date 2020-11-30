import logging


def preprocessing(data, ctx):
	logging.info("Preprocessing for workflow smtest")
	if data:
		image = data[0].get("data") or data[0].get("body")
		return [image]
	else:
		return


def postprocessing(data, ctx):
	logging.info("Postprocessing for workflow smtest")
	if data:
		data = data[0].get("body")
		return [data]
	else:
		return

