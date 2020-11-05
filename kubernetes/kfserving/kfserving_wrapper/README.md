# KFServing Wrapper

The KFServing wrapper folder contains three files :

1) __main__.py
2) TorchserveModel.py
3) TSModelRepository.py

The KFServing wrapper files were created to enable the Torchserve integration with KFServing. 

1) the __main__.py file parses the model snapshot from the config.properties present in the KFServing side and passes the parameters like inference address, management address and the model address to the KFServing side to handle the input request and response. 


2) The TorchserveModel.py file contains the methods to handle the request and response that comes from the Torchserve side and passes it on to the KFServing side.

3) TSModelRepository.py file contains the intialize method for the parameters that gets passed on to the Torchservemodel.py. 


