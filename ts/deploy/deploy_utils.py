from string import Template


HANDLER_TEMPLATE=Template("""
from ${base_class_module} import ${base_class}

class ${handler_name}(${base_class}):
${init}
${preprocess}
${inference}
${postprocess}
""")

def deployable(base_class, **kwargs):
    print(f"{kwargs=}")
    def inner(func, **inner_kwargs):
        func.__is_deployable__ = True
        func._kwargs = kwargs
        func._base_class = base_class
        
        return func
    
    return inner


def codegen_handler_from_deployable(deployable):
    assert hasattr(deployable, "__is_deployable__")
    assert hasattr(deployable, "_kwargs")
    assert hasattr(deployable, "_base_class")
    
    from types import MethodType
    from inspect import getsource, getmodule, getmodulename
    
    handler = deployable._base_class()
    
    handler.inference = MethodType(deployable, handler)
    
    for f_name in ["initialize", "preprocess", "postprocess"]:
        if f_name in deployable._kwargs:
            setattr(handler, f_name, MethodType(deployable._kwargs[f_name], handler))
            
    def correct_indentation(source):
        import textwrap
        source = textwrap.dedent(source)
        source = textwrap.indent(source, '    ')
        return source
    
    inference_source = getsource(deployable)
    import re
    pattern = r'\@deployable\([^)]+\)'
    inference_source = re.sub(pattern, '', inference_source)
    
    sub_dict = {
        "handler_name": "CustomHandler",
        "base_class": deployable._base_class.__name__,
        "base_class_module": getmodule(deployable._base_class).__name__,
        "inference": correct_indentation(inference_source),
    }
    for f_name in ["init", "initialize", "preprocess", "postprocess"]:
        func = deployable._kwargs.get(f_name, None)
        sub_dict[f_name] = correct_indentation(getsource(func)) if func else ""
        
    handler_code = HANDLER_TEMPLATE.substitute(**sub_dict)
            
    return handler_code
    


def deploy(deployable):
    print(codegen_handler_from_deployable(deployable))
    
    
    
    